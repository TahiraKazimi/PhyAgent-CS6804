"""
PhyAgent: Planner-Generator-Evaluator pipeline for physics-grounded video generation.

Pipeline:
    prompt --> PlannerAgent --> reference trees {T*^(i)}
    prompt --> GeneratorAgent (Wan2.1-14B) --> video v_hat
    (v_hat, {T*^(i)}) --> EvaluatorAgent --> structured verdict
    if not accepted: feed verdict back to GeneratorAgent for regeneration
"""

import os
import re
import json
import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple

from stage_reward_multi import StageAwareReward, MultiProcessRewardResult


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class ReferenceTree:
    """Planner output for one physical principle."""
    principle_name: str
    physics_domain: str
    stages: List[Dict]   # [{"stage": 1, "name": ..., "description": ..., "children": [...]}]
    depth: int


@dataclass
class GlobalVerdict:
    hps: float = 0.0
    pc: float = 0.0
    sa: float = 0.0
    score: float = 0.0


@dataclass
class PrincipleVerdict:
    """Local verdict for one physical principle."""
    principle_name: str
    present: bool
    alignment: float
    completeness: float
    ordering: float
    r_hier: float
    failure_mode: Optional[str] = None   # "absent", "incomplete", "out_of_order", None


@dataclass
class EvaluatorVerdict:
    """Structured output the evaluator returns to the generator."""
    accept: bool
    total_score: float
    local_verdicts: List[PrincipleVerdict]
    global_verdict: GlobalVerdict
    reason: str
    regenerate_hint: Optional[str] = None


@dataclass
class PipelineResult:
    final_video: object
    final_verdict: EvaluatorVerdict
    history: List[EvaluatorVerdict] = field(default_factory=list)
    reference_trees: List[ReferenceTree] = field(default_factory=list)
    attempts: int = 0


# ============================================================================
# Planner Agent
# ============================================================================

PLANNER_SYSTEM = """You are a physics planner. Given a prompt for a video, identify
the distinct physical principles entailed by the prompt and decompose each one into a
hierarchical, temporally-ordered tree of stages.

Rules:
- A "physical principle" is a single coherent physics process (e.g., rigid-body fall,
  fluid splash, combustion, buoyancy). Identify all that the prompt entails.
- For each principle, produce 2-5 outer stages in chronological order.
- A stage may recursively expand into 2-4 sub-stages if the dynamics warrant it
  (max depth 3). Only expand when the sub-events are themselves temporally ordered.
- Be conservative: only include stages that *must* occur for the principle to be
  physically faithful. Do not invent decorative stages.

Respond with ONLY this JSON, no prose:
````json
{
  "principles": [
    {
      "principle_name": "snake_case_name",
      "physics_domain": "rigid_body | fluid | thermal | acoustic | optical | electromagnetic | other",
      "stages": [
        {
          "stage": 1,
          "name": "stage_name",
          "description": "what physically happens",
          "children": [
            {"stage": 1, "name": "...", "description": "...", "children": []}
          ]
        }
      ]
    }
  ]
}
```"""


class PlannerAgent:
    """Decomposes a prompt into reference trees, one per principle."""

    def __init__(self, llm_model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 device: str = "cuda", shared_vlm=None, shared_processor=None):
        self.device = device
        self.llm_model_path = llm_model_path
        # Allow sharing VLM weights with the evaluator's perception module
        self._llm = shared_vlm
        self._proc = shared_processor

    def _load(self):
        if self._llm is not None:
            return
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self._llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.llm_model_path, torch_dtype=torch.float16, device_map=self.device,
        )
        self._proc = AutoProcessor.from_pretrained(self.llm_model_path)
        self._llm.eval()

    @staticmethod
    def _tree_depth(stages: List[Dict]) -> int:
        if not stages:
            return 0
        return 1 + max(
            (PlannerAgent._tree_depth(s.get("children", [])) for s in stages),
            default=0,
        )

    def plan(self, prompt: str, max_retries: int = 3) -> List[ReferenceTree]:
        self._load()
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": f'Prompt: "{prompt}"'},
        ]
        text_in = self._proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._proc(text=[text_in], padding=True, return_tensors="pt").to(self.device)

        for _ in range(max_retries):
            with torch.no_grad():
                out = self._llm.generate(
                    **inputs, max_new_tokens=2048, temperature=0.3, do_sample=True
                )
            gen = out[0][inputs.input_ids.shape[1]:]
            text = self._proc.decode(gen, skip_special_tokens=True)
            trees = self._parse(text)
            if trees:
                return trees
        return []

    def _parse(self, text: str) -> List[ReferenceTree]:
        m = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        raw = m.group(1) if m else text
        raw = re.sub(r',\s*([}\]])', r'\1', raw)
        try:
            data = json.loads(raw[raw.find('{'):raw.rfind('}') + 1])
        except json.JSONDecodeError:
            return []

        out = []
        for p in data.get("principles", []):
            stages = p.get("stages", [])
            if not stages:
                continue
            out.append(ReferenceTree(
                principle_name=p["principle_name"],
                physics_domain=p.get("physics_domain", "other"),
                stages=stages,
                depth=self._tree_depth(stages),
            ))
        return out

    @staticmethod
    def to_gt_entry(trees: List[ReferenceTree]) -> Dict:
        """Convert planner output into the GT format StageAwareReward expects."""
        # Flatten hierarchical stages into the flat format used by the existing
        # reward code. (The recursive Hungarian variant would consume children directly.)
        processes = []
        for t in trees:
            flat_stages = []
            def walk(node, prefix=""):
                name = f"{prefix}{node['name']}" if prefix else node['name']
                flat_stages.append({
                    "stage": len(flat_stages) + 1,
                    "name": name,
                    "description": node["description"],
                })
                for c in node.get("children", []):
                    walk(c, prefix=f"{name} > ")
            for s in t.stages:
                walk(s)
            processes.append({
                "process_name": t.principle_name,
                "physics_domain": t.physics_domain,
                "stages": flat_stages,
            })
        return {"processes": processes}


# ============================================================================
# Generator Agent (Wan2.1-14B)
# ============================================================================

class GeneratorAgent:
    """Wraps Wan2.1-14B and accepts evaluator feedback for regeneration."""

    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        device: str = "cuda",
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        guidance_scale: float = 6.0,
        num_inference_steps: int = 35,
        flow_shift: float = 3.0,
    ):
        self.model_id = model_id
        self.device = device
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.flow_shift = flow_shift
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        from diffusers import WanPipeline, UniPCMultistepScheduler
        print(f"Loading Wan2.1-14B from {self.model_id}...")
        self._pipe = WanPipeline.from_pretrained(self.model_id, torch_dtype=torch.bfloat16)
        self._pipe.scheduler = UniPCMultistepScheduler.from_config(
            self._pipe.scheduler.config, flow_shift=self.flow_shift
        )
        self._pipe.to(self.device)

    def _augment_prompt(self, prompt: str, hint: Optional[str], trees: List[ReferenceTree]) -> str:
        """Inject planner stages and evaluator feedback into the generation prompt."""
        parts = [prompt]
        if trees:
            stage_lines = []
            for t in trees:
                top_stages = ", ".join(s["name"] for s in t.stages)
                stage_lines.append(f"{t.principle_name}: {top_stages}")
            parts.append(
                "Show the following physical processes unfolding in chronological order. "
                + "; ".join(stage_lines) + "."
            )
        if hint:
            parts.append(f"Important: {hint}")
        return " ".join(parts)

    def generate(
        self,
        prompt: str,
        reference_trees: List[ReferenceTree],
        regenerate_hint: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self._load()
        full_prompt = self._augment_prompt(prompt, regenerate_hint, reference_trees)
        gen = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None

        with torch.no_grad():
            out = self._pipe(
                prompt=full_prompt,
                height=self.height,
                width=self.width,
                num_frames=self.num_frames,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                generator=gen,
            )
        # diffusers returns frames as list of PIL images; stack to (T,H,W,C) np.uint8
        raw = out.frames[0]
        # Handle all the shapes diffusers might return
        if isinstance(raw, list):
            # list of PIL images
            if hasattr(raw[0], "convert"):
                frames = np.stack([np.array(f.convert("RGB")) for f in raw], axis=0)
            else:
                # list of arrays
                frames = np.stack([np.asarray(f) for f in raw], axis=0)
        elif isinstance(raw, torch.Tensor):
            frames = raw.detach().cpu().numpy()
        else:
            frames = np.asarray(raw)

        # Normalize to uint8 (T, H, W, C)
        if frames.dtype != np.uint8:
            if frames.max() <= 1.0 + 1e-3:
                frames = (frames.clip(0, 1) * 255).round().astype(np.uint8)
            else:
                frames = frames.clip(0, 255).astype(np.uint8)

        # If channels-first (T, C, H, W), transpose
        if frames.ndim == 4 and frames.shape[1] in (1, 3) and frames.shape[-1] not in (1, 3):
            frames = frames.transpose(0, 2, 3, 1)
        return frames

    def to_cpu(self):
        if self._pipe is not None:
            self._pipe.to("cpu")
            torch.cuda.empty_cache()


# ============================================================================
# Evaluator Agent
# ============================================================================

class EvaluatorAgent:
    """
    Local + global judgment over a generated video.

    Local: wraps StageAwareReward, which extracts an observed tree from v_hat
    via the perception VLM and aligns it (Hungarian) against the planner's
    reference trees.

    Global: HPSv2 + VideoPhy2 (PC, SA), all VLM-based.
    """

    def __init__(
        self,
        stage_reward: StageAwareReward,
        videophy_model_path: str = "videophysics/videocon_physics",
        hps_version: str = "v2.1",
        # Aggregation weights
        w_local: float = 0.5,
        w_hps: float = 0.15,
        w_pc: float = 0.20,
        w_sa: float = 0.15,
        # Acceptance thresholds
        accept_total: float = 0.65,
        accept_coverage: float = 0.75,
        accept_pc: float = 0.5,
        accept_sa: float = 0.5,
    ):
        self.stage_reward = stage_reward
        self.videophy_model_path = videophy_model_path
        self.hps_version = hps_version
        self.w_local = w_local
        self.w_hps = w_hps
        self.w_pc = w_pc
        self.w_sa = w_sa
        self.accept_total = accept_total
        self.accept_coverage = accept_coverage
        self.accept_pc = accept_pc
        self.accept_sa = accept_sa
        self._hps = None
        self._videophy = None

    # ---------------- global terms ----------------

    def _score_hps(self, frames: np.ndarray, prompt: str) -> float:
        if self._hps is None:
            import hpsv2
            self._hps = hpsv2
        # Sample T frames uniformly
        idxs = np.linspace(0, len(frames) - 1, 8, dtype=int)
        scores = []
        for i in idxs:
            frame = frames[i]
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0 + 1e-3:
                    frame = (frame.clip(0, 1) * 255).round().astype(np.uint8)
                else:
                    frame = frame.clip(0, 255).astype(np.uint8)
            img = Image.fromarray(frame)
            s = self._hps.score(img, prompt, hps_version=self.hps_version)
            scores.append(float(np.asarray(s).mean()))
        return float(np.clip(np.mean(scores), 0.0, 1.0))

    def _score_videophy(self, frames: np.ndarray, prompt: str) -> Tuple[float, float]:
        """Returns (PC, SA) using VideoPhy2's VLM judge."""
        # Reuse the stage_reward VLM as a stand-in for VideoPhy2 if not loaded separately.
        # In production you would load videophy_model_path explicitly.
        self.stage_reward._load_vlm()
        pil_frames = self.stage_reward._extract_frames(frames)
        prompt_text = (
            f'Caption: "{prompt}"\n\n'
            "Rate the video on two axes from 0 to 5:\n"
            "1. PC = physical commonsense: motion obeys real-world physics.\n"
            "2. SA = semantic alignment: video depicts the caption.\n"
            'Respond ONLY as JSON: {"pc": <0-5>, "sa": <0-5>}'
        )
        content = [{"type": "image", "image": img} for img in pil_frames]
        content.append({"type": "text", "text": prompt_text})
        msgs = [{"role": "user", "content": content}]
        proc = self.stage_reward._vlm_processor
        text_in = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text_in], images=pil_frames, padding=True,
                      return_tensors="pt").to(self.stage_reward._vlm.device)

        with torch.no_grad():
            out = self.stage_reward._vlm.generate(
                **inputs, max_new_tokens=128, temperature=0.2, do_sample=True
            )
        gen = out[0][inputs.input_ids.shape[1]:]
        text = proc.decode(gen, skip_special_tokens=True)
        raw = self.stage_reward._extract_outermost_json(text)
        if not raw:
            return 0.0, 0.0
        try:
            d = json.loads(re.sub(r',\s*([}\]])', r'\1', raw))
            return float(d.get("pc", 0)) / 5.0, float(d.get("sa", 0)) / 5.0
        except Exception:
            return 0.0, 0.0

    # ---------------- main entry ----------------
    def evaluate(
            self,
            video: np.ndarray,
            prompt: str,
            reference_trees: List[ReferenceTree],
        ) -> EvaluatorVerdict:
            log = print if getattr(self, "verbose", True) else (lambda *a, **k: None)

            log("\n" + "=" * 70)
            log(f"[Evaluator] prompt: {prompt!r}")
            log(f"[Evaluator] reference principles: {[t.principle_name for t in reference_trees]}")
            log("=" * 70)

            # Local: align observed tree against planner's reference trees
            gt_entry = PlannerAgent.to_gt_entry(reference_trees)
            local: MultiProcessRewardResult = self.stage_reward.score(
                video=video, gt_entry=gt_entry, caption=prompt
            )

            log(f"\n[Local] total_reward={local.total_reward:.3f}  "
                f"process_coverage={local.process_coverage:.2f}  "
                f"absent={local.absent_processes}")

            # Per-principle verdicts with diagnosed failure modes
            local_verdicts = []
            for pr in local.process_rewards:
                failure_mode = None
                if not pr.present:
                    failure_mode = "absent"
                elif pr.completeness_score < 0.5:
                    failure_mode = "incomplete"
                elif pr.ordering_score < 0.5:
                    failure_mode = "out_of_order"
                local_verdicts.append(PrincipleVerdict(
                    principle_name=pr.process_name,
                    present=pr.present,
                    alignment=pr.alignment_score,
                    completeness=pr.completeness_score,
                    ordering=pr.ordering_score,
                    r_hier=pr.process_reward,
                    failure_mode=failure_mode,
                ))
                flag = f"  [!] {failure_mode}" if failure_mode else ""
                log(f"  - {pr.process_name:<25s}  "
                    f"r_hier={pr.process_reward:.3f}  "
                    f"align={pr.alignment_score:.3f}  "
                    f"comp={pr.completeness_score:.3f}  "
                    f"order={pr.ordering_score:.3f}{flag}")

            # Global
            log("\n[Global] scoring...")
            hps = self._score_hps(video, prompt)
            pc, sa = self._score_videophy(video, prompt)
            gscore = self.w_hps * hps + self.w_pc * pc + self.w_sa * sa
            gscore /= max(self.w_hps + self.w_pc + self.w_sa, 1e-8)
            global_v = GlobalVerdict(hps=hps, pc=pc, sa=sa, score=gscore)
            log(f"[Global] HPS={hps:.3f}  PC={pc:.3f}  SA={sa:.3f}  score={gscore:.3f}")

            # Aggregate total score
            total = self.w_local * local.total_reward + (1 - self.w_local) * gscore
            total = float(np.clip(total, 0.0, 1.0))

            accept, reason, hint = self._decide(total, local, local_verdicts, global_v)

            log("\n[Verdict] " + ("ACCEPT" if accept else "REJECT") +
                f"  total={total:.3f}  "
                f"(local={local.total_reward:.3f}, global={gscore:.3f})")
            log(f"[Verdict] reason: {reason}")
            if hint:
                log(f"[Verdict] regenerate_hint: {hint}")
            log("=" * 70 + "\n")

            return EvaluatorVerdict(
                accept=accept,
                total_score=total,
                local_verdicts=local_verdicts,
                global_verdict=global_v,
                reason=reason,
                regenerate_hint=hint,
            )
    # def evaluate(
    #     self,
    #     video: np.ndarray,
    #     prompt: str,
    #     reference_trees: List[ReferenceTree],
    # ) -> EvaluatorVerdict:
    #     # Local: align observed tree against planner's reference trees
    #     gt_entry = PlannerAgent.to_gt_entry(reference_trees)
    #     local: MultiProcessRewardResult = self.stage_reward.score(
    #         video=video, gt_entry=gt_entry, caption=prompt
    #     )

    #     # Per-principle verdicts with diagnosed failure modes
    #     local_verdicts = []
    #     for pr in local.process_rewards:
    #         failure_mode = None
    #         if not pr.present:
    #             failure_mode = "absent"
    #         elif pr.completeness_score < 0.5:
    #             failure_mode = "incomplete"
    #         elif pr.ordering_score < 0.5:
    #             failure_mode = "out_of_order"
    #         local_verdicts.append(PrincipleVerdict(
    #             principle_name=pr.process_name,
    #             present=pr.present,
    #             alignment=pr.alignment_score,
    #             completeness=pr.completeness_score,
    #             ordering=pr.ordering_score,
    #             r_hier=pr.process_reward,
    #             failure_mode=failure_mode,
    #         ))

    #     # Global
    #     hps = self._score_hps(video, prompt)
    #     pc, sa = self._score_videophy(video, prompt)
    #     gscore = self.w_hps * hps + self.w_pc * pc + self.w_sa * sa
    #     gscore /= max(self.w_hps + self.w_pc + self.w_sa, 1e-8)
    #     global_v = GlobalVerdict(hps=hps, pc=pc, sa=sa, score=gscore)

    #     # Aggregate total score
    #     total = self.w_local * local.total_reward + (1 - self.w_local) * gscore
    #     total = float(np.clip(total, 0.0, 1.0))

    #     accept, reason, hint = self._decide(total, local, local_verdicts, global_v)
    #     return EvaluatorVerdict(
    #         accept=accept,
    #         total_score=total,
    #         local_verdicts=local_verdicts,
    #         global_verdict=global_v,
    #         reason=reason,
    #         regenerate_hint=hint,
    #     )

    def _decide(self, total, local, lvs, gv):
        if (
            total >= self.accept_total
            and local.process_coverage >= self.accept_coverage
            and gv.pc >= self.accept_pc
            and gv.sa >= self.accept_sa
        ):
            return True, "all thresholds met", None

        problems, hints = [], []
        absent = [v.principle_name for v in lvs if v.failure_mode == "absent"]
        if absent:
            problems.append(f"missing principles: {absent}")
            hints.append(f"explicitly show {', '.join(absent)} unfolding over time")

        incomplete = [v.principle_name for v in lvs if v.failure_mode == "incomplete"]
        if incomplete:
            problems.append(f"incomplete stages in: {incomplete}")
            hints.append(f"show all stages of {', '.join(incomplete)}")

        ooo = [v.principle_name for v in lvs if v.failure_mode == "out_of_order"]
        if ooo:
            problems.append(f"out-of-order stages in: {ooo}")
            hints.append(f"keep chronological order in {', '.join(ooo)}")

        if gv.pc < self.accept_pc:
            problems.append(f"low physical commonsense (PC={gv.pc:.2f})")
            hints.append("ensure motion obeys gravity, momentum, and material behavior")
        if gv.sa < self.accept_sa:
            problems.append(f"weak prompt alignment (SA={gv.sa:.2f})")
            hints.append("stay closer to the literal prompt content")

        return False, "; ".join(problems) or "below threshold", " ".join(hints) or None


# ============================================================================
# Orchestrator
# ============================================================================

class PhyAgent:
    """End-to-end planner -> generator -> evaluator pipeline."""

    def __init__(
        self,
        planner: PlannerAgent,
        generator: GeneratorAgent,
        evaluator: EvaluatorAgent,
        max_attempts: int = 3,
        seed_base: int = 0,
    ):
        self.planner = planner
        self.generator = generator
        self.evaluator = evaluator
        self.max_attempts = max_attempts
        self.seed_base = seed_base

    def run(self, prompt: str) -> PipelineResult:
        # 1. Plan reference trees once per prompt
        trees = self.planner.plan(prompt)
        if not trees:
            raise RuntimeError(f"Planner produced no reference trees for prompt: {prompt}")

        history = []
        last_video = None
        last_verdict = None
        hint = None

        for attempt in range(self.max_attempts):
            # 2. Generate (with feedback if this is a retry)
            video = self.generator.generate(
                prompt=prompt,
                reference_trees=trees,
                regenerate_hint=hint,
                seed=self.seed_base + attempt,
            )

            # 3. Evaluate
            verdict = self.evaluator.evaluate(video, prompt, trees)
            history.append(verdict)
            last_video, last_verdict = video, verdict

            if verdict.accept:
                return PipelineResult(
                    final_video=video, final_verdict=verdict,
                    history=history, reference_trees=trees, attempts=attempt + 1,
                )
            hint = verdict.regenerate_hint

        # 4. No attempt accepted: return best by total_score
        best_idx = int(np.argmax([v.total_score for v in history]))
        return PipelineResult(
            final_video=last_video,
            final_verdict=history[best_idx],
            history=history,
            reference_trees=trees,
            attempts=len(history),
        )


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    stage_reward = StageAwareReward(
        vlm_model_path="Qwen/Qwen2.5-VL-7B-Instruct",
        embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cuda:0",
    )
    planner = PlannerAgent(device="cuda:0",
                           shared_vlm=stage_reward._vlm,
                           shared_processor=stage_reward._vlm_processor)
    generator = GeneratorAgent(device="cuda:1")
    evaluator = EvaluatorAgent(stage_reward=stage_reward)

    pipeline = PhyAgent(planner, generator, evaluator, max_attempts=3)
    result = pipeline.run("a rubber ball is dropped from a height bouncing as it contacts the floor ")

    print(f"Attempts: {result.attempts}, accept: {result.final_verdict.accept}")
    print(f"Total: {result.final_verdict.total_score:.3f}")
    for v in result.final_verdict.local_verdicts:
        print(f"  {v.principle_name}: r_hier={v.r_hier:.3f}, failure={v.failure_mode}")

