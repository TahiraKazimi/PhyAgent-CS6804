"""
Stage-Aware Physics Reward Model — Multi-Process (v2)
=====================================================

Given GT processes (from generate_stages_from_prompts.py), extracts
per-process stages from generated video via Qwen2.5-VL, then scores
each process independently.

Key design: the VLM prompt explicitly lists the GT process names so
the output is 1:1 with GT. If a process is completely absent from the
video, the VLM returns empty stages for it → reward = 0 for that process.

GT format:
    {
        "processes": [
            {
                "process_name": "glass_resonance",
                "physics_domain": "acoustics",
                "stages": [{"stage": 1, "name": "...", "description": "..."}]
            },
            ...
        ]
    }

Usage:
    reward_model = StageAwareReward(
        vlm_model_path="Qwen/Qwen2.5-VL-7B-Instruct",
        embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    # Full reward with per-process breakdown
    result = reward_model.score(video, gt_entry, caption="...")
    # result.total_reward, result.process_rewards[i].process_reward, etc.

    # Fast scalar for GRPO
    reward = reward_model.score_fast(video, gt_entry, caption="...")

    # Skip VLM — use pre-extracted stages
    result = reward_model.score_preextracted(gt_entry, gen_entry)
"""

import os
import re
import json
import base64
import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Union, Dict, Tuple
from dataclasses import dataclass, field


# ============================================================================
# Prompts
# ============================================================================

PROCESS_EXTRACTION_PROMPT = """Watch this video carefully. It is supposed to show: "{caption}"

The video should contain the following distinct physical processes:
{process_list}

For EACH process listed above, describe the physical stages that actually occur in the video in chronological order.
- If a process is clearly present, list its stages (2-5 stages each).
- If a process is completely ABSENT or not observable, return an empty stages list for it.
- Do NOT fabricate stages you cannot see. Be honest about what is and isn't present.

You MUST return exactly one entry per process listed above, using the exact process names provided.

Respond in this exact JSON format and nothing else:
```json
{{
    "processes": [
        {{
            "process_name": "exact_name_from_list_above",
            "present": true,
            "stages": [
                {{"stage": 1, "name": "stage_name", "description": "what physically happens"}},
                {{"stage": 2, "name": "stage_name", "description": "what physically happens"}}
            ]
        }},
        {{
            "process_name": "exact_name_from_list_above",
            "present": false,
            "stages": []
        }}
    ]
}}
```"""


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class ProcessRewardResult:
    """Reward result for a single process."""
    process_name: str = ""
    present: bool = False          # VLM says this process is in the video
    alignment_score: float = 0.0
    completeness_score: float = 0.0
    ordering_score: float = 0.0
    stage_ratio: float = 0.0      # min(gen,gt)/max(gen,gt) — penalizes count mismatch
    raw_reward: float = 0.0       # before stage_ratio penalty
    process_reward: float = 0.0   # final = raw_reward * stage_ratio
    gt_stages: List[Dict] = field(default_factory=list)
    gen_stages: List[Dict] = field(default_factory=list)
    stage_matches: List[Dict] = field(default_factory=list)


@dataclass
class MultiProcessRewardResult:
    """Aggregate reward across all processes."""
    total_reward: float = 0.0
    process_coverage: float = 0.0   # fraction of GT processes present
    process_rewards: List[ProcessRewardResult] = field(default_factory=list)
    absent_processes: List[str] = field(default_factory=list)


# ============================================================================
# Main Class
# ============================================================================

class StageAwareReward:

    def __init__(
        self,
        vlm_model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
        num_frames: int = 8,
        frame_size: int = 384,
        # Stage-level threshold
        stage_match_threshold: float = 0.68,
        # Reward weights (within each process)
        w_align: float = 0.4,
        w_complete: float = 0.35,
        w_order: float = 0.25,
        # Cross-process aggregation
        w_process_coverage: float = 0.2,
    ):
        self.vlm_model_path = vlm_model_path
        self.embed_model_name = embed_model_name
        self.device = device
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.stage_match_threshold = stage_match_threshold
        self.w_align = w_align
        self.w_complete = w_complete
        self.w_order = w_order
        self.w_process_coverage = w_process_coverage

        self._vlm = None
        self._vlm_processor = None
        self._embed_model = None

    # ----------------------------------------------------------------
    # Model loading
    # ----------------------------------------------------------------

    def _load_vlm(self):
        if self._vlm is not None:
            return
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        print(f"Loading VLM: {self.vlm_model_path}...")
        self._vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.vlm_model_path, torch_dtype=torch.float16, device_map=self.device,
        )
        self._vlm_processor = AutoProcessor.from_pretrained(self.vlm_model_path)
        self._vlm.eval()
        print("VLM loaded.")

    def _load_embed_model(self):
        if self._embed_model is not None:
            return
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {self.embed_model_name}...")
        self._embed_model = SentenceTransformer(self.embed_model_name, device=self.device)
        print("Embedding model loaded.")

    # ----------------------------------------------------------------
    # Frame extraction
    # ----------------------------------------------------------------

    def _extract_frames(self, video) -> List[Image.Image]:
        if isinstance(video, str):
            return self._extract_frames_from_path(video)
        if isinstance(video, torch.Tensor):
            video = video.detach().cpu().numpy()
        if isinstance(video, np.ndarray):
            if video.ndim == 5:
                video = video[0]
            if video.max() <= 1.0:
                video = (video * 255).astype(np.uint8)
            else:
                video = video.astype(np.uint8)
            T = video.shape[0]
            indices = np.linspace(0, T - 1, self.num_frames, dtype=int)
            return [Image.fromarray(video[i]).resize((self.frame_size, self.frame_size)) for i in indices]
        raise TypeError(f"Unsupported video type: {type(video)}")

    def _extract_frames_from_path(self, video_path: str) -> List[Image.Image]:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.frame_size, self.frame_size))
            frames.append(Image.fromarray(frame))
        cap.release()
        return frames

    # ----------------------------------------------------------------
    # Stage extraction via VLM
    # ----------------------------------------------------------------

    def extract_processes(
        self,
        video: Union[str, torch.Tensor, np.ndarray],
        caption: str,
        gt_process_names: List[str],
        max_retries: int = 3,
    ) -> Dict[str, Dict]:
        """
        Extract per-process stages from video using Qwen2.5-VL.
        The prompt explicitly lists gt_process_names so VLM output
        is 1:1 with GT.

        Returns:
            dict mapping process_name -> {"present": bool, "stages": [...]}
        """
        self._load_vlm()
        frames = self._extract_frames(video)
        if not frames:
            return {name: {"present": False, "stages": []} for name in gt_process_names}

        # Build process list for prompt
        process_list = "\n".join(f"  {i+1}. {name}" for i, name in enumerate(gt_process_names))
        text_prompt = PROCESS_EXTRACTION_PROMPT.format(
            caption=caption, process_list=process_list
        )

        content = [{"type": "image", "image": img} for img in frames]
        content.append({"type": "text", "text": text_prompt})
        messages = [{"role": "user", "content": content}]

        text_input = self._vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._vlm_processor(
            text=[text_input], images=frames, padding=True, return_tensors="pt",
        ).to(self._vlm.device)

        for attempt in range(max_retries):
            with torch.no_grad():
                output_ids = self._vlm.generate(
                    **inputs, max_new_tokens=1536, temperature=0.3, do_sample=True,
                )
            generated = output_ids[0][inputs.input_ids.shape[1]:]
            text = self._vlm_processor.decode(generated, skip_special_tokens=True)
            parsed = self._parse_processes(text, gt_process_names)
            if parsed is not None:
                return parsed

        # All retries failed — return empty for all
        return {name: {"present": False, "stages": []} for name in gt_process_names}

    # ----------------------------------------------------------------
    # Parsing
    # ----------------------------------------------------------------

    @staticmethod
    def _extract_outermost_json(text: str) -> Optional[str]:
        start = text.find('{')
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == '\\' and in_string:
                escape = True
                continue
            if c == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
        return None

    def _parse_processes(
        self, text: str, expected_names: List[str]
    ) -> Optional[Dict[str, Dict]]:
        """
        Parse VLM output into {process_name: {"present": bool, "stages": [...]}}.
        Maps VLM output back to expected_names by position or exact name match.
        """
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        raw = self._extract_outermost_json(
            json_match.group(1) if json_match else text
        )
        if not raw:
            return None

        raw = re.sub(r',\s*([}\]])', r'\1', raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None

        # Extract process list from response
        if 'processes' not in data or not isinstance(data['processes'], list):
            return None

        proc_list = data['processes']

        # Validate each process entry
        for proc in proc_list:
            if not isinstance(proc, dict) or 'process_name' not in proc:
                return None
            if 'stages' not in proc:
                proc['stages'] = []
            if not isinstance(proc['stages'], list):
                return None
            # Validate individual stages (only for non-empty)
            for s in proc['stages']:
                if not isinstance(s, dict) or 'name' not in s or 'description' not in s:
                    return None

        # Build output dict keyed by expected GT names
        result = {}

        # First pass: exact name match
        gen_by_name = {}
        for proc in proc_list:
            gen_by_name[proc['process_name']] = proc

        matched_gen = set()
        for name in expected_names:
            if name in gen_by_name:
                p = gen_by_name[name]
                present = p.get('present', len(p['stages']) > 0)
                result[name] = {"present": present, "stages": p['stages']}
                matched_gen.add(name)

        # Second pass: positional fallback for unmatched
        # (VLM might rephrase names but keep the same order)
        if len(result) < len(expected_names):
            unmatched_expected = [n for n in expected_names if n not in result]
            unmatched_gen_procs = [p for p in proc_list if p['process_name'] not in matched_gen]

            for exp_name, gen_proc in zip(unmatched_expected, unmatched_gen_procs):
                present = gen_proc.get('present', len(gen_proc['stages']) > 0)
                result[exp_name] = {"present": present, "stages": gen_proc['stages']}

        # Fill any still-missing with absent
        for name in expected_names:
            if name not in result:
                result[name] = {"present": False, "stages": []}

        return result

    # ----------------------------------------------------------------
    # Embedding
    # ----------------------------------------------------------------

    def _embed_stages(self, stages: List[Dict]) -> np.ndarray:
        """Embed stage descriptions. Returns (N, D) L2-normalized."""
        self._load_embed_model()
        texts = [
            f"{s['name'].replace('_', ' ')}: {s['description']}"
            for s in stages
        ]
        return np.array(self._embed_model.encode(texts, normalize_embeddings=True))

    # ----------------------------------------------------------------
    # Per-process reward components
    # ----------------------------------------------------------------

    def compute_alignment_dtw(
        self, gt_emb: np.ndarray, gen_emb: np.ndarray
    ) -> Tuple[float, List[Tuple[int, int]]]:
        """
        DTW alignment score between GT and generated stages.
        Non-diagonal moves (skip/repeat) incur a penalty so that
        1 gen stage matching 3 GT stages can't score high.
        """
        N, M = gt_emb.shape[0], gen_emb.shape[0]
        if N == 0 or M == 0:
            return 0.0, []

        dist = 1.0 - (gt_emb @ gen_emb.T)  # (N, M), values in [0, 2]

        # Penalty for non-diagonal moves: set to 1.0 (= max cosine distance)
        # so skipping a GT stage is as bad as a completely wrong match.
        # With 1 gen vs 3 GT, DTW must take 2 vertical skips → cost >= 2.0
        skip_penalty = 1.0

        C = np.full((N + 1, M + 1), np.inf)
        C[0, 0] = 0.0
        backtrack = {}

        for i in range(1, N + 1):
            for j in range(1, M + 1):
                # Diagonal: proper 1:1 alignment
                diag = C[i - 1, j - 1] + dist[i - 1, j - 1]
                # Vertical: GT stage i skipped (no gen stage matched to it)
                vert = C[i - 1, j] + skip_penalty
                # Horizontal: gen stage j reused (already matched a previous GT stage)
                horiz = C[i, j - 1] + skip_penalty

                candidates = [(diag, (i - 1, j - 1)), (vert, (i - 1, j)), (horiz, (i, j - 1))]
                best_cost, best_prev = min(candidates, key=lambda x: x[0])
                C[i, j] = best_cost
                backtrack[(i, j)] = best_prev

        path = []
        i, j = N, M
        while (i, j) != (0, 0):
            if i > 0 and j > 0:
                path.append((i - 1, j - 1))
            prev = backtrack.get((i, j))
            if prev is None:
                break
            i, j = prev
        path.reverse()

        # Normalize by number of GT stages (not max(N,M))
        score = 1.0 - min(C[N, M] / N, 1.0)
        return float(max(score, 0.0)), path

    def compute_completeness(
        self, gt_emb: np.ndarray, gen_emb: np.ndarray
    ) -> Tuple[float, List[Dict]]:
        """
        Fraction of GT stages with a generated match above threshold.
        Uses 1:1 greedy matching — each gen stage can only be claimed
        by one GT stage, so 1 gen stage can't satisfy 3 GT stages.
        """
        N, M = gt_emb.shape[0], gen_emb.shape[0]
        if M == 0:
            return 0.0, []

        sim = gt_emb @ gen_emb.T  # (N, M)

        # Greedy 1:1: pick highest sim pairs, no gen reuse
        matches = [None] * N
        used_gen = set()

        # Sort all (gt_i, gen_j) pairs by descending similarity
        pairs = []
        for i in range(N):
            for j in range(M):
                pairs.append((float(sim[i, j]), i, j))
        pairs.sort(reverse=True)

        for s, i, j in pairs:
            if i in {m["gt_stage_idx"] for m in matches if m is not None}:
                continue  # this GT already matched
            if j in used_gen:
                continue  # this gen already claimed
            is_match = s >= self.stage_match_threshold
            matches[i] = {
                "gt_stage_idx": i,
                "best_gen_idx": j,
                "similarity": round(s, 4),
                "matched": is_match,
            }
            if is_match:
                used_gen.add(j)

        # Fill any GT stages that got no match at all
        matched_count = 0
        for i in range(N):
            if matches[i] is None:
                # Find best remaining gen
                best_j = int(np.argmax(sim[i]))
                best_sim = float(sim[i, best_j])
                matches[i] = {
                    "gt_stage_idx": i,
                    "best_gen_idx": best_j,
                    "similarity": round(best_sim, 4),
                    "matched": False,
                }
            if matches[i]["matched"]:
                matched_count += 1

        return matched_count / N if N > 0 else 0.0, matches

    def compute_ordering(self, matches: List[Dict]) -> float:
        """Check if matched stages preserve GT ordering."""
        indices = [m["best_gen_idx"] for m in matches if m["matched"]]
        if not indices:
            return 0.0
        if len(indices) == 1:
            return 1.0
        correct = sum(1 for a, b in zip(indices[:-1], indices[1:]) if a < b)
        return correct / (len(indices) - 1)

    # ----------------------------------------------------------------
    # Single-process reward
    # ----------------------------------------------------------------

    def _score_single_process(
        self,
        gt_stages: List[Dict],
        gen_stages: List[Dict],
        process_name: str = "",
        present: bool = True,
    ) -> ProcessRewardResult:
        """Compute reward for one process: GT stages vs generated stages."""
        if not gen_stages or not present:
            return ProcessRewardResult(
                process_name=process_name,
                present=present,
                gt_stages=gt_stages,
                gen_stages=gen_stages,
            )

        gt_emb = self._embed_stages(gt_stages)
        gen_emb = self._embed_stages(gen_stages)

        alignment, _ = self.compute_alignment_dtw(gt_emb, gen_emb)
        completeness, matches = self.compute_completeness(gt_emb, gen_emb)
        ordering = self.compute_ordering(matches)

        # Stage count ratio penalty: if gen has far fewer stages than GT,
        # the raw scores get scaled down. E.g. 1 gen vs 3 GT → ratio = 0.33
        N, M = len(gt_stages), len(gen_stages)
        stage_ratio = min(M, N) / max(M, N) if max(M, N) > 0 else 0.0

        raw_reward = (
            self.w_align * alignment
            + self.w_complete * completeness
            + self.w_order * ordering
        )
        process_reward = raw_reward * stage_ratio

        return ProcessRewardResult(
            process_name=process_name,
            present=True,
            alignment_score=alignment,
            completeness_score=completeness,
            ordering_score=ordering,
            stage_ratio=stage_ratio,
            raw_reward=float(raw_reward),
            process_reward=float(np.clip(process_reward, 0.0, 1.0)),
            gt_stages=gt_stages,
            gen_stages=gen_stages,
            stage_matches=matches,
        )

    # ----------------------------------------------------------------
    # Multi-process reward aggregation
    # ----------------------------------------------------------------

    def _compute_reward(
        self,
        gt_processes: List[Dict],
        gen_processes: Dict[str, Dict],
    ) -> MultiProcessRewardResult:
        """
        Compute per-process rewards and aggregate.

        Args:
            gt_processes: list of {"process_name": str, "stages": [...]}
            gen_processes: dict mapping process_name -> {"present": bool, "stages": [...]}
                           (keyed by GT process names — 1:1 correspondence)
        """
        process_rewards = []
        absent = []
        n_present = 0

        for gt_proc in gt_processes:
            name = gt_proc['process_name']
            gen = gen_processes.get(name, {"present": False, "stages": []})

            result = self._score_single_process(
                gt_stages=gt_proc['stages'],
                gen_stages=gen['stages'],
                process_name=name,
                present=gen.get('present', len(gen['stages']) > 0),
            )
            process_rewards.append(result)

            if result.present:
                n_present += 1
            else:
                absent.append(name)

        # Coverage: fraction of GT processes present in video
        n_gt = len(gt_processes)
        process_coverage = n_present / n_gt if n_gt > 0 else 0.0

        # Aggregate: mean of per-process rewards (0 for absent processes)
        all_rewards = [pr.process_reward for pr in process_rewards]
        mean_process_reward = np.mean(all_rewards) if all_rewards else 0.0

        # Blend mean per-process reward with coverage bonus
        total = (
            (1.0 - self.w_process_coverage) * mean_process_reward
            + self.w_process_coverage * process_coverage
        )

        return MultiProcessRewardResult(
            total_reward=float(np.clip(total, 0.0, 1.0)),
            process_coverage=process_coverage,
            process_rewards=process_rewards,
            absent_processes=absent,
        )

    # ----------------------------------------------------------------
    # Normalize GT format
    # ----------------------------------------------------------------

    @staticmethod
    def _normalize_gt(gt_entry: Union[Dict, List[Dict]]) -> List[Dict]:
        """
        Accept:
          - {"processes": [{"process_name": ..., "stages": [...]}]}
          - [{"stage": 1, "name": ..., "description": ...}]
          - {"stages": [...]}
        Always returns list of {"process_name": str, "stages": [...]}.
        """
        if isinstance(gt_entry, list):
            return [{"process_name": "combined", "stages": gt_entry}]
        if isinstance(gt_entry, dict):
            if 'processes' in gt_entry and isinstance(gt_entry['processes'], list):
                return gt_entry['processes']
            if 'stages' in gt_entry and isinstance(gt_entry['stages'], list):
                return [{"process_name": "combined", "stages": gt_entry['stages']}]
        raise ValueError(f"Cannot normalize GT entry: {type(gt_entry)}")

    @staticmethod
    def _normalize_gen_to_dict(
        gen_entry: Union[Dict, List[Dict]], expected_names: List[str]
    ) -> Dict[str, Dict]:
        """
        Normalize gen_entry into {process_name: {"present": bool, "stages": [...]}}
        for use with score_preextracted.
        """
        if isinstance(gen_entry, list):
            # Flat stage list → single combined process
            return {"combined": {"present": True, "stages": gen_entry}}

        if isinstance(gen_entry, dict):
            if 'processes' in gen_entry and isinstance(gen_entry['processes'], list):
                result = {}
                gen_list = gen_entry['processes']
                # Exact name match first
                gen_by_name = {p['process_name']: p for p in gen_list}
                matched = set()
                for name in expected_names:
                    if name in gen_by_name:
                        p = gen_by_name[name]
                        result[name] = {
                            "present": p.get('present', len(p.get('stages', [])) > 0),
                            "stages": p.get('stages', []),
                        }
                        matched.add(name)

                # Positional fallback
                unmatched_exp = [n for n in expected_names if n not in result]
                unmatched_gen = [p for p in gen_list if p['process_name'] not in matched]
                for exp_name, gen_proc in zip(unmatched_exp, unmatched_gen):
                    result[exp_name] = {
                        "present": gen_proc.get('present', len(gen_proc.get('stages', [])) > 0),
                        "stages": gen_proc.get('stages', []),
                    }

                # Fill missing
                for name in expected_names:
                    if name not in result:
                        result[name] = {"present": False, "stages": []}

                return result

            if 'stages' in gen_entry and isinstance(gen_entry['stages'], list):
                return {"combined": {"present": True, "stages": gen_entry['stages']}}

        raise ValueError(f"Cannot normalize gen entry: {type(gen_entry)}")

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def score(
        self,
        video: Union[str, torch.Tensor, np.ndarray],
        gt_entry: Union[Dict, List[Dict]],
        caption: str = "",
    ) -> MultiProcessRewardResult:
        """
        Full reward: tell VLM which processes to look for → extract stages → score.

        Args:
            video: path, tensor (T,H,W,C) or (B,T,H,W,C), or numpy
            gt_entry: GT data with "processes" list
            caption: original prompt

        Returns:
            MultiProcessRewardResult with per-process breakdown
        """
        gt_processes = self._normalize_gt(gt_entry)
        gt_names = [p['process_name'] for p in gt_processes]

        gen_processes = self.extract_processes(
            video, caption=caption, gt_process_names=gt_names
        )
        return self._compute_reward(gt_processes, gen_processes)

    def score_fast(
        self,
        video: Union[str, torch.Tensor, np.ndarray],
        gt_entry: Union[Dict, List[Dict]],
        caption: str = "",
    ) -> float:
        """Fast scalar reward for GRPO training."""
        return self.score(video, gt_entry, caption).total_reward

    def score_preextracted(
        self,
        gt_entry: Union[Dict, List[Dict]],
        gen_entry: Union[Dict, List[Dict]],
    ) -> MultiProcessRewardResult:
        """
        Compute reward from pre-extracted stages (no VLM inference).
        """
        gt_processes = self._normalize_gt(gt_entry)
        gt_names = [p['process_name'] for p in gt_processes]
        gen_dict = self._normalize_gen_to_dict(gen_entry, gt_names)
        return self._compute_reward(gt_processes, gen_dict)

    def batch_score(
        self,
        videos: List[Union[str, torch.Tensor, np.ndarray]],
        gt_entries: List[Union[Dict, List[Dict]]],
        captions: Optional[List[str]] = None,
    ) -> List[MultiProcessRewardResult]:
        """Batch reward for GRPO."""
        if captions is None:
            captions = [""] * len(videos)
        return [
            self.score(video, gt, caption)
            for video, gt, caption in zip(videos, gt_entries, captions)
        ]

    def batch_score_fast(
        self,
        videos: List[Union[str, torch.Tensor, np.ndarray]],
        gt_entries: List[Union[Dict, List[Dict]]],
        captions: Optional[List[str]] = None,
    ) -> List[float]:
        """Batch scalar rewards for GRPO."""
        return [r.total_reward for r in self.batch_score(videos, gt_entries, captions)]

    # ----------------------------------------------------------------
    # Device management
    # ----------------------------------------------------------------

    def to_device(self, device: str):
        self.device = device
        if self._vlm is not None:
            self._vlm.to(device)
        if self._embed_model is not None:
            self._embed_model = self._embed_model.to(device)

    def to_cpu(self):
        self.to_device("cpu")
        torch.cuda.empty_cache()