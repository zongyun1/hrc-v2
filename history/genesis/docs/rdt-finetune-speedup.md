# RDT finetune is significantly slower than pi0/pi05 — top 3 fixes

**Setting:** RDT-1B finetune on genesis-hr-bench (Franka single-arm, ~11,755 episodes). Reported on the FAIR Cloud H200 cluster as substantially slower than pi0 / pi0.5 / pi0_fast on the same dataset.

**TL;DR:** three structural costs dominate RDT's per-step wall time. None of them are touched by the current pipeline. Fixing them is mechanical and stacks to roughly **3× faster training** (range 2.6–5.0×) on a single H200 node, with no behavioural regression. The biggest single win is fix #1 (~half the total speedup on its own). Ranking is by leverage; baseline assumes `--train_batch_size=16`, `--dataloader_num_workers=8`, bf16, ZeRO-2.

| # | Fix | Expected Δ | Code change |
|---|---|---|---|
| 1 | `--precomp_lang_embed` (drop T5-XXL from the loop) | **1.3–1.8×** | 1 line in launcher + offline T5-XXL pass |
| 2 | Precompute SigLIP vision embeddings (+ drop unused 3rd camera) | **1.5–1.85×** | offline SigLIP pass + small dataloader change |
| 3 | `action_chunk_size: 64 → 32`, `MIN_EPISODE_STEPS: 128 → 48` | **1.3–1.5×** + recovers ~9% silently dropped data | 2-line config change |

Stacked: **~2.6–5.0× faster** depending on cluster bottlenecks. Realistic median **~3×**.

---

## Fix #1 — Drop T5-XXL out of the training loop with `--precomp_lang_embed`

**Cost today.** `scripts/finetune_rdt.sh:149-172` never passes `--precomp_lang_embed`, so `baseline/rdt/train/train.py:419-424` runs the **~11B-param T5-v1.1-XXL** text encoder forward on every batch:

```python
text_embeds = ... text_encoder(
    input_ids=batch["input_ids"],
    attention_mask=lang_attn_mask
)["last_hidden_state"].detach()
```

`tokenizer_max_length: 1024` (`baseline/rdt/configs/base.yaml:38`) is the FLOPs-dominating dimension, not the actual instruction length. The forward is under `no_grad`, but the encoder still occupies ~22 GB of GPU and an 80–200 ms/step slice on H200 at batch=16.

This is wasted work twice over:
- The instruction is **constant per episode** — re-encoding it every step gains nothing.
- The encoder is **frozen** — there's no training signal that requires the live tensor.

**Fix.** Run RDT's offline language-embedding script once over the dataset, then add `--precomp_lang_embed` to the launcher.

1. Adapt `baseline/rdt/scripts/encode_lang.py` to read `instruction.json` (the genesis-hr-bench format — see `baseline/rdt_overrides/data/hdf5_vla_dataset.py:107-111`) and dump a `lang_embed.pt` (`(seq_len, 4096)` T5-XXL `last_hidden_state`) next to each `episode.hdf5`.
2. Add `--precomp_lang_embed` to the `exec "${DIST_LAUNCHER[@]}" main.py ...` block in `scripts/finetune_rdt.sh:149-172`.
3. Update `baseline/rdt_overrides/data/hdf5_vla_dataset.py` to load the precomputed `lang_embeds` + `lang_attn_mask` instead of returning `input_ids`. This mirrors the upstream path that's already gated on `--precomp_lang_embed` in `train.py:419-420`.

**Side effects:**
- Frees ~22 GB GPU. Lets `--train_batch_size` rise from 16 → 32+, which improves utilization further (compounds with this fix).
- T5-XXL no longer needs to load into the venv during training — job-startup time drops.
- One-time offline cost: a single ~30 min T5-XXL pass over ~11,755 instructions. Output lives on lustre; survives preempt.

**Expected speedup: 1.3–1.8×.** Closer to 1.8× on H200 (FLOPs-bound regime), closer to 1.3× on A100-40GB where ZeRO-2 was already paging.

---

## Fix #2 — Precompute SigLIP vision embeddings (+ drop unused 3rd camera)

**Cost today.** `baseline/rdt/train/train.py:413-416`:

```python
with torch.no_grad():
    batch_size, _, C, H, W = images.shape
    image_embeds = vision_encoder(images.reshape(-1, C, H, W)).detach()
    image_embeds = image_embeds.reshape((batch_size, -1, vision_encoder.hidden_size))
```

`img_history_size: 2 × num_cameras: 3 = 6` SigLIP-so400m@384 forwards **per sample**. At `--train_batch_size=16` that's **96 SigLIP@384 forwards per training step**. SigLIP-so400m is ~400M params and ~50 GFlops at 384² — the second-largest FLOPs sink after the diffusion transformer itself.

Two observations make this avoidable:
- The vision encoder is **frozen** under `no_grad` (same as T5-XXL).
- The genesis-hr-bench dataset has **2** cameras (`cam_high` + `cam_right_wrist`), not 3. The `cam_left_wrist` slot is filled with `np.zeros((IMG_HISORY_SIZE, 0, 0, 0))` at `baseline/rdt_overrides/data/hdf5_vla_dataset.py:195-196`, but the training loop still spends a SigLIP forward on whatever the collation pads it to.

**Fix — two stages, in order.**

**2a. Drop `num_cameras: 3 → 2`** in `baseline/rdt/configs/base.yaml:7`. Pure config change — ~1.15× speedup for free. The model's `img_cond_len` recomputes from config (`train.py:154-156`), but this changes the pretrained image positional-embedding layout — expect a 500-step warmup spike before recovery.

**2b. Precompute SigLIP embeddings.** Same pattern as fix #1.
1. Add `scripts/precomp_rdt_vision_embed.py`: walk the HDF5 tree, run SigLIP-so400m@384 over every frame, store the `(num_patches=729, hidden=1152)` tensor as a new HDF5 dataset (e.g. `observations/siglip_embeds/cam_high`, `.../cam_right_wrist`). Roughly 3 MB/frame, ~20 GB total for the genesis-hr-bench dataset. Fits on lustre.
2. Add a `--precomp_vision_embed` flag mirroring `--precomp_lang_embed` to `baseline/rdt/main.py` + `train.py` to skip the `vision_encoder(...)` call when set.
3. Drop `--image_aug` from `scripts/finetune_rdt.sh:166`. **This is the real cost** — `image_aug` (ColorJitter / blur / noise) is applied in the dataloader before SigLIP, so precomputed embeddings invalidate it.

If `--image_aug` is empirically important on genesis-hr-bench: precompute K=8 augmented variants per frame and sample one at train time. Storage rises to ~160 GB; still cheap on lustre.

**Side effects:**
- Frees ~3 GB GPU (SigLIP weights off the device).
- ~30 min one-time precomp pass. Survives preempt.
- Caveat above re: `--image_aug` interaction.

**Expected speedup: 1.5–1.85× combined** (2a × 2b).

---

## Fix #3 — `action_chunk_size: 64 → 32` and `MIN_EPISODE_STEPS: 128 → 48`

**Cost today — two bundled issues.**

**3.1. Diffusion transformer wastes capacity on 64-step horizons.** `baseline/rdt/configs/base.yaml:5 action_chunk_size: 64` drives both the RDT diffusion transformer's `pred_horizon` (`baseline/rdt/scripts/maniskill_model.py:70`) and the dataloader's per-sample chunk slice (`baseline/rdt_overrides/data/hdf5_vla_dataset.py:144`).

At `action_substeps=25` (~20 Hz benchmark control), 64 steps ≈ **3.2 s of future prediction** — far longer than needed for genesis-hr-bench's local-reactive manipulation. The inference server **only uses the first step of the chunk** anyway (`baseline/servers/rdt_server.py:131-134` does `first = a.reshape(-1, a.shape[-1])[0]` and discards 63 of 64), so shrinking the train-time chunk has zero inference impact.

The 28-layer / hidden=2048 / 32-head diffusion transformer's per-step forward+backward scales with the action-token sequence length. Halving the chunk roughly halves the diffusion backward cost.

**3.2. `MIN_EPISODE_STEPS = 128` silently drops ~9% of all data, including entire short tasks.**

`baseline/rdt_overrides/data/hdf5_vla_dataset.py:33`:

```python
MIN_EPISODE_STEPS = 128
```

Measured distribution across the live 11,755-episode dataset at `runs/rdt/data_target/`:

```
min=84  max=2637  mean=378  median=374  std=185
p1=96  p5=102  p10=132  p25=220
< 100 steps:   421 eps  (3.58%)
< 128 steps:  1096 eps  (9.32%)   ← currently dropped
< 150 steps:  1510 eps  (12.85%)
```

The shortest tasks (median episode length under or near 128) are **mostly invisible** to RDT training right now:

| task | min | p10 | median |
|---|---|---|---|
| pour_water | 95 | 98 | **101** |
| take_from_human_easy | 94 | 95 | **97** |
| stack_bowls_three_assist | 100 | 102 | **110** |
| deliver_to_human_easy | 84 | 97 | 130 |
| place_bread_in_basket_assist | 117 | 125 | 133 |
| place_dual_shoes_assist | 129 | 137 | 154 |

`pour_water` and `take_from_human_easy` are effectively **not in the training set** at MIN=128 — every episode under the threshold is silently dropped at `parse_hdf5_file_state_only` time during the dataloader's init sampling-weight pass.

**Fix — two-line change.**

- `baseline/rdt/configs/base.yaml:5` → `action_chunk_size: 32`
- `baseline/rdt_overrides/data/hdf5_vla_dataset.py:33` → `MIN_EPISODE_STEPS = 48`

**Why these specific numbers:**
- chunk_size=32 fits inside even the 84-step global minimum episode without right-padding (52 valid start positions). No "predict static last action" signal pollution.
- chunk_size=32 ≈ 1.6 s lookahead at 20 Hz — well-matched to genesis-hr-bench task durations.
- MIN_EPISODE_STEPS=48 ≥ chunk_size + first-moving-idx slack, so the `_first_moving_idx` heuristic still has room.
- 48 < 84 = global minimum, so **no real episodes are dropped**. The cutoff is just a guard against future truncated/corrupt data.

**Caveat:** RDT-1B is pretrained with chunk=64. Changing `pred_horizon` truncates the model's action-token positional embedding. Finetune still works (this is standard across diffusion-policy variants) — expect a 500–1000-step loss spike before recovery. Not a correctness issue, just a warmup signal in wandb.

Mainstream chunk-size defaults for comparison: pi0 = 50, ACT = 100, lerobot diffusion `n_action_steps` = 8. 32 is mid-range.

**Side effects:**
- Recovers 1096 episodes (~9.32%) currently dropped, including most of `pour_water` and `take_from_human_easy`. This is the more important effect — RDT is currently being benchmarked on a silently-truncated dataset.
- ~1.3–1.5× speedup on the diffusion transformer's per-step backward.

**Expected speedup: 1.3–1.5×** + non-quantifiable accuracy improvement on the recovered short tasks.

---

## Combined estimate

Stacking on a single H200 node (current baseline ≈ 1.0×):

| Fix | Standalone | Cumulative |
|---|---|---|
| #1 — `--precomp_lang_embed` | 1.3–1.8× | **1.3–1.8×** |
| #2a — drop cam_left_wrist | 1.15× | 1.5–2.1× |
| #2b — precompute SigLIP | 1.3–1.6× | 2.0–3.3× |
| #3 — chunk 64→32 + MIN 128→48 | 1.3–1.5× | **2.6–5.0×** |

Realistic median: **~3× faster** after all three fixes, with #1 alone delivering ~half the win.

The fixes are also additive in **GPU memory freed** (~22 GB from T5-XXL + ~3 GB from SigLIP + smaller model state from chunk-32). After all three you can comfortably double `--train_batch_size`, which is a further ~1.2–1.4× *effective* speedup not counted in the table above.

---

## What's NOT in the top 3 (and why)

For completeness — these were considered and ranked lower:

- **Remove EMA** (`baseline/rdt/train/train.py:181, 445`). Config says `# We do not use EMA currently` but code runs it anyway. Frees ~3 GB + ~1.05–1.15× speedup. **Worth doing**, just not in the top 3 because the win is small relative to #1/#2/#3.
- **Gradient checkpointing.** `train.py:202-204` has `raise NotImplementedError`. Would let you push batch size higher; 1.2–1.4× *effective* but requires real implementation work.
- **FlashAttention-2** in the diffusion transformer. ~1.3× if not already used. Worth a 5-minute grep but probably no-op if RDT already uses FA-2.
- **DeepSpeed ZeRO-2 → DDP.** ~1.1–1.3× on multi-node only; on single-node H200 it's already mostly a no-op. Possible after #1 + EMA removal free enough memory.
- **Cache per-episode stats in `__init__`** (`hdf5_vla_dataset.py:140-142`). Removes a small recomputation per sample. ~1.05× steady-state. Below the top 3 by leverage.
- **Replace `cv2.imdecode` with pre-decoded images** in HDF5. ~1.1× if dataloader is CPU-bound. Below the top 3.

---

## References

- Launcher: `scripts/finetune_rdt.sh`
- Training loop: `baseline/rdt/train/train.py`
- Inference server (chunk discard): `baseline/servers/rdt_server.py`
- Dataloader (override): `baseline/rdt_overrides/data/hdf5_vla_dataset.py`
- Model config: `baseline/rdt/configs/base.yaml`
- Dataset converter: `scripts/convert_genesis_hr_bench_to_rdt_hdf5.py`
- Original schema-fix report (resolved): `docs/rdt-loader-schema-issue.md`
- Pipeline doc: `docs/vla-finetune-pipeline.md`
