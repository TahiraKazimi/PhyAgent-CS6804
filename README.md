# PhyAgent

**Hierarchical Physical Alignment via a Planner–Generator–Evaluator Pipeline**

PhyAgent is an agentic pipeline for physics-grounded text-to-video generation. Instead of producing a single uninterpretable score, PhyAgent decomposes each prompt into its constituent physical principles, generates a candidate video, and returns a structured verdict that names *which* principles failed and *how* — closing the loop by feeding that verdict back into the generator for a targeted retry.

## Overview

A single forward pass cannot diagnose whether a generated video is physically faithful, which physical principle it violated, or where in the temporal unfolding the failure occurred. PhyAgent addresses this by structuring physics evaluation as an interaction between three specialized agents:

```
prompt ──► PlannerAgent ──► reference trees {T*^(i)}
prompt ──► GeneratorAgent (Wan2.1) ──► candidate video v̂
(v̂, {T*^(i)}) ──► EvaluatorAgent ──► structured verdict
                                         │
                                         └─► if rejected: regenerate_hint ──► GeneratorAgent
```

- **PlannerAgent** decomposes the prompt into the physical principles it entails (e.g., *rigid-body fall*, *fluid splash*) and produces a temporally-ordered, hierarchical reference tree of stages for each one.
- **GeneratorAgent** produces a candidate video using a backbone text-to-video diffusion model (Wan2.1 by default), conditioned on both the prompt and the planner's stage descriptions.
- **EvaluatorAgent** judges the video along two complementary axes: a **local** stage-level judgment per principle (alignment, completeness, ordering), and a **global** video-level judgment (HPSv2 + VideoPhy2 PC/SA). When the verdict is rejection, the evaluator emits a structured hint that names the failed principle and its failure mode (`absent`, `incomplete`, `out_of_order`), which the generator consumes on the next attempt.

## Repository structure

```
.
├── phyAgents.py              # All agents + orchestrator
├── stage_reward_multi.py     # StageAwareReward: hierarchical local scoring
└── requirements_physAgent.txt
```

The local evaluator is built on top of `StageAwareReward` (`stage_reward_multi.py`), which performs the per-principle stage extraction via VLM and the Hungarian alignment between reference and observed trees.

## Installation

```bash
conda create -n phyagent python=3.11 -y
conda activate phyagent
pip install -r requirements_physAgent.txt
```

Tested with PyTorch 2.8.0 + CUDA 12.8 on H100/H200 GPUs. Python 3.13 is **not** recommended — HPSv2 has known packaging issues on 3.13.

If `hpsv2` fails to find its BPE vocab file, reinstall from source:

```bash
pip uninstall -y hpsv2
pip install git+https://github.com/tgxs002/HPSv2.git
```

## Quick start

```python
from stage_reward_multi import StageAwareReward
from phyAgents import PlannerAgent, GeneratorAgent, EvaluatorAgent, PhyAgent

# Shared VLM perception for planner + evaluator stage extraction
stage_reward = StageAwareReward(
    vlm_model_path="Qwen/Qwen2.5-VL-7B-Instruct",
    embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda:0",
)

planner   = PlannerAgent(device="cuda:0",
                         shared_vlm=stage_reward._vlm,
                         shared_processor=stage_reward._vlm_processor)
generator = GeneratorAgent(device="cuda:1")          # Wan2.1 on a separate GPU
evaluator = EvaluatorAgent(stage_reward=stage_reward)

pipeline = PhyAgent(planner, generator, evaluator, max_attempts=3)
result   = pipeline.run("a rubber ball is dropped from a height bouncing as it contacts the floor")

print(f"Attempts: {result.attempts}, accepted: {result.final_verdict.accept}")
print(f"Total score: {result.final_verdict.total_score:.3f}")
for v in result.final_verdict.local_verdicts:
    print(f"  {v.principle_name}: r_hier={v.r_hier:.3f}, failure={v.failure_mode}")
```

Run directly:

```bash
python phyAgents.py
```

## How each agent works

### PlannerAgent

Produces a list of `ReferenceTree` objects. Each tree has a `principle_name`, a `physics_domain` (rigid_body, fluid, thermal, etc.), and a list of temporally-ordered `stages`, where each stage may recursively expand into up to 3 levels of children. The planner uses a Qwen2.5-VL backbone with a structured-JSON system prompt and retries on parse failures.

By default the planner shares VLM weights with the evaluator's perception module — pass `shared_vlm` and `shared_processor` to avoid loading two copies of Qwen2.5-VL. To use independent weights, omit these arguments.

### GeneratorAgent

Wraps a `diffusers` text-to-video pipeline (default: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`; swap to `Wan-AI/Wan2.1-T2V-14B-Diffusers` for higher quality if VRAM allows). The agent augments the user's prompt in two ways:

1. The planner's stage names are appended as a chronological hint (e.g. `"Show the following physical processes unfolding in chronological order. ball_fall: release, freefall, impact; ball_bounce: rebound, decay, rest."`).
2. On retry attempts, the evaluator's `regenerate_hint` is appended (e.g. `"Important: explicitly show fluid_splash unfolding over time"`).

Configurable via constructor: `height`, `width`, `num_frames`, `guidance_scale`, `num_inference_steps`, `flow_shift`. Returns frames as `np.uint8` arrays of shape `(T, H, W, C)`.

### EvaluatorAgent

Returns an `EvaluatorVerdict` with:

- `accept` — boolean, true if all thresholds are met.
- `total_score` — weighted blend of local and global scores in `[0, 1]`.
- `local_verdicts` — per-principle `PrincipleVerdict` with `alignment`, `completeness`, `ordering`, `r_hier`, and a diagnosed `failure_mode` (`absent` / `incomplete` / `out_of_order` / `None`).
- `global_verdict` — `GlobalVerdict` with `hps`, `pc`, `sa`, and aggregate `score`.
- `reason` — human-readable summary of failures.
- `regenerate_hint` — actionable feedback for the generator's next attempt.

**Local scoring** is delegated to `StageAwareReward.score`, which extracts an observed tree from the video via Qwen2.5-VL and aligns it against the planner's reference tree using Hungarian assignment, with stage matches gated by a similarity threshold.

**Global scoring** combines:
- `HPSv2` for visual quality and aesthetic alignment (averaged over uniformly sampled frames).
- `VideoPhy2` (or a stand-in VLM judge) for physical commonsense (PC) and semantic alignment (SA), each rated 0–5 and normalized to `[0, 1]`.

Default weights: `w_local=0.5`, `w_hps=0.15`, `w_pc=0.20`, `w_sa=0.15`. Default acceptance thresholds: `accept_total=0.65`, `accept_coverage=0.75`, `accept_pc=0.5`, `accept_sa=0.5`. **Calibrate these on a held-out set** before reporting numbers — the defaults are reasonable starting points, not validated operating points.

## Example terminal output

```
======================================================================
[Evaluator] prompt: 'a rubber ball is dropped from a height bouncing as it contacts the floor'
[Evaluator] reference principles: ['ball_fall', 'ball_bounce']
======================================================================

[Local] total_reward=0.512  process_coverage=1.00  absent=[]
  - ball_fall                  r_hier=0.731  align=0.812  comp=0.750  order=1.000
  - ball_bounce                r_hier=0.293  align=0.554  comp=0.333  order=0.500  [!] incomplete

[Global] scoring...
[Global] HPS=0.612  PC=0.600  SA=0.700  score=0.633

[Verdict] REJECT  total=0.572  (local=0.512, global=0.633)
[Verdict] reason: incomplete stages in: ['ball_bounce']
[Verdict] regenerate_hint: show all stages of ball_bounce
======================================================================
```

## GPU placement

The pipeline loads three VLM/diffusion components. A typical placement on two H100s:

| Component                | Device   | Approx. VRAM (bf16/fp16) |
| ------------------------ | -------- | ------------------------ |
| Qwen2.5-VL-7B (shared)   | cuda:0   | ~14 GB                   |
| Wan2.1-T2V-14B           | cuda:1   | ~28 GB                   |
| HPSv2                    | cuda:0   | ~2 GB                    |
| Sentence-Transformers    | cuda:0   | ~0.5 GB                  |

For tighter setups, call `generator.to_cpu()` between generation and evaluation, and `stage_reward.to_cpu()` after evaluation.

## Configuration reference

### `PhyAgent(planner, generator, evaluator, max_attempts=3, seed_base=0)`
Top-level orchestrator. `max_attempts` caps the retry loop; if no attempt is accepted, the highest-scoring attempt is returned.

### `PlannerAgent(llm_model_path, device, shared_vlm=None, shared_processor=None)`
`shared_vlm`/`shared_processor` reuse the evaluator's VLM weights to save memory.

### `GeneratorAgent(model_id, device, height, width, num_frames, guidance_scale, num_inference_steps, flow_shift)`
Any `diffusers`-compatible Wan T2V checkpoint works. Frame count is bounded by Wan's training resolution.

### `EvaluatorAgent(stage_reward, ..., w_local, w_hps, w_pc, w_sa, accept_total, accept_coverage, accept_pc, accept_sa)`
All weights and thresholds are constructor arguments.


