# pi0 / pi05 pretrained-load failures on genesis-hr-bench

When loading the official pretrained pi0 / pi05 checkpoints from `lerobot/...` on
HuggingFace and fine-tuning on the
[`zhouqh/hrbench`](https://huggingface.co/datasets/zhouqh/hrbench/tree/main/lerobot_datasets)
LeRobot v3.0 datasets, the training run dies before it ever takes a step. Two
distinct errors were seen depending on which pretrained repo we point at. This
doc captures both failures with reproducers, root-causes, and the open questions
we need zhouqh to answer to unblock fine-tuning from pretrained.

## TL;DR

**Resolved 2026-05-23** — both errors are unblocked by switching to
`pi0_base` / `pi05_base` and passing `--rename_map` to map our 2 cameras
onto 2 of the model's 3 expected slots. pi0's built-in
`missing_img_keys` path zero-masks the 3rd. No code changes, no dataset
rebuild. See [Resolution](#resolution-2026-05-23) below. Sanity smokes
PASS for both pi0 and pi05 (200 steps each, A100, jobs 58275353 / 58275921).

| Pretrained repo | Failure | Root cause | Fix |
|---|---|---|---|
| `lerobot/pi0` | `draccus.utils.DecodingError: ... fields are not valid for PI0Config` (8 stale fields in HF `config.json`) | The HF repo's `config.json` was published against an older `lerobot` version whose `PI0Config` dataclass had different fields than `lerobot v0.5.2` (what's installed here). | Switch to `lerobot/pi0_base` (zero stale fields). |
| `lerobot/pi0_base`, `lerobot/pi05_base` | `ValueError: Feature mismatch between dataset/environment and policy config.` (model expects 3 cameras, dataset has 2) | Pretrained pi0_base / pi05_base were trained on a **3-camera setup** (`base_0_rgb` + `left_wrist_0_rgb` + `right_wrist_0_rgb`). `zhouqh/hrbench` LeRobot v3.0 datasets only ship **2 cameras** (`observation.image`, `observation.wrist_image`). | Pass `--rename_map` (skips the validator) and let pi0's modeling fill the missing 3rd camera with `-1` + zero attention mask. |

From-scratch fine-tune (no `--pretrained`) works fine on these datasets — verified
end-to-end by running and successfully completing pi0/pi05 sanity jobs (e.g.,
`1371243`, `1371244`) and full jobs (e.g., `1371258`, `1371259`).

## Environment

- Cluster: FAIR Cloud h200 partition, 8 × H200 / node.
- Repo: `/storage/home/zengh/projects/genesis-hr-bench` @ commit `829b6c8`.
- LeRobot install: `/home/zengh/envs/lerobot/` (Python 3.12.13).
- `lerobot` source: vendored at `baseline/lerobot/src/lerobot/` (codebase_version `v3.0`, package version `v0.5.2`).
- Data root: `HF_LEROBOT_HOME=/lustre-storage/datasets/zengh/huggingface/lerobot`.
- Datasets in use (downloaded from `zhouqh/hrbench:/lerobot_datasets/`):
  - `genesis-hr-bench/genesis_hr_bench_lerobot_v30/`    — EE control, state/action shape `[7]`.
  - `genesis-hr-bench/genesis_hr_bench_lerobot_qpos_v30/`   — qpos delta, state/action shape `[8]`.
  - `genesis-hr-bench/genesis_hr_bench_lerobot_qpos_abs_v30/` — qpos absolute, shape `[8]`.
  - All three have the same 2 cameras: `observation.image` (head) + `observation.wrist_image` (single wrist), 224×224 RGB, 20 fps.

## Error 1 — stale config schema (`lerobot/pi0`)

### Reproducer

```bash
cd /storage/home/zengh/projects/genesis-hr-bench
# Sanity-mode (1 GPU, 200 steps) to fail fast:
scripts/finetune.sh --model pi0 --slurm \
    --gpus-per-node 1 --sanity --time 2:00:00 \
    --pretrained lerobot/pi0

# (failed job id seen on FAIR: 1371245, exited in 39s with exit code 1)
```

The driver `scripts/finetune_lerobot_v2.sh` translates `--pretrained X` into
`--policy.path=X` for lerobot's `lerobot_train`. The effective `lerobot_train`
launch:

```
accelerate launch --num_processes=1 -m lerobot.scripts.lerobot_train \
    --policy.type=pi0 --policy.dtype=bfloat16 \
    --policy.path=lerobot/pi0 \
    --dataset.repo_id=genesis-hr-bench/genesis_hr_bench_lerobot_v30 \
    ... [other defaults]
```

### Exact error

```
draccus.utils.DecodingError: The fields
  `resize_imgs_with_padding`, `adapt_to_pi_aloha`, `use_delta_joint_actions_aloha`,
  `proj_width`, `num_steps`, `use_cache`, `attention_implementation`, `train_state_proj`
are not valid for PI0Config
```

### Diagnosis

```python
from huggingface_hub import hf_hub_download
import json
cfg = json.load(open(hf_hub_download("lerobot/pi0", "config.json")))
stale = ["resize_imgs_with_padding", "adapt_to_pi_aloha", "use_delta_joint_actions_aloha",
         "proj_width", "num_steps", "use_cache", "attention_implementation", "train_state_proj"]
print([k for k in stale if k in cfg])
# -> all 8 fields present (lerobot/pi0)

cfg_base = json.load(open(hf_hub_download("lerobot/pi0_base", "config.json")))
print([k for k in stale if k in cfg_base])
# -> [] (lerobot/pi0_base has no stale fields)
```

`lerobot/pi0` is published for an older `PI0Config` (probably ≤ lerobot 0.3.x).
`lerobot/pi0_base` matches `lerobot v0.5.2`. Switching to `lerobot/pi0_base`
resolves this error but exposes Error 2.

## Error 2 — feature mismatch (3-camera model vs 2-camera dataset)

### Reproducer

```bash
cd /storage/home/zengh/projects/genesis-hr-bench
# Full-mode (8 GPU) since the error fires at policy construction:
scripts/finetune.sh --model pi0 --slurm \
    --nodes 1 --gpus-per-node 8 --full \
    --time 3-00:00:00 --mem 0 \
    --exp-name "pi0_ee_repro" \
    --pretrained lerobot/pi0_base
# (failed job id seen on FAIR: 1371256, exited in 3m33s with exit code 1)

# Same shape of failure for pi05:
scripts/finetune.sh --model pi05 --slurm \
    --nodes 1 --gpus-per-node 8 --full \
    --time 3-00:00:00 --mem 0 \
    --exp-name "pi05_ee_repro" \
    --pretrained lerobot/pi05_base
# (failed job id seen on FAIR: 1371255, exited in 5m34s with exit code 1)
```

Effective `lerobot_train` launch (pi0 case):

```
accelerate launch --multi_gpu --num_processes=8 -m lerobot.scripts.lerobot_train \
    --policy.type=pi0 --policy.dtype=bfloat16 \
    --policy.path=lerobot/pi0_base \
    --dataset.repo_id=genesis-hr-bench/genesis_hr_bench_lerobot_v30 \
    ... [other defaults]
```

### Exact error

```
ValueError: Feature mismatch between dataset/environment and policy config.
- Missing features: ['observation.images.base_0_rgb',
                     'observation.images.left_wrist_0_rgb',
                     'observation.images.right_wrist_0_rgb']
- Extra features:   ['observation.image',
                     'observation.wrist_image']

Please ensure your dataset and policy use consistent feature names.
If your dataset uses different observation keys (e.g., cameras named differently),
use the `--rename_map` argument, for example:
  --rename_map='{"observation.images.left": "observation.images.camera1",
                 "observation.images.top":  "observation.images.camera2"}'
```

Raised at `baseline/lerobot/src/lerobot/policies/utils.py:213` inside
`validate_visual_features_consistency` called from
`baseline/lerobot/src/lerobot/policies/factory.py:558`.

### Diagnosis

```python
# Pretrained model expects 3 cameras (per its config.json):
from huggingface_hub import hf_hub_download
import json
cfg = json.load(open(hf_hub_download("lerobot/pi0_base", "config.json")))
img_keys = [k for k in cfg.get("input_features", {}) if "image" in k.lower()]
print(img_keys)
# -> ['observation.images.base_0_rgb',
#     'observation.images.left_wrist_0_rgb',
#     'observation.images.right_wrist_0_rgb']

# zhouqh's dataset has 2 cameras (per its meta/info.json):
import json
ds_info = json.load(open(
    "/lustre-storage/datasets/zengh/huggingface/lerobot/"
    "genesis-hr-bench/genesis_hr_bench_lerobot_v30/meta/info.json"))
print([k for k in ds_info["features"] if "image" in k.lower()])
# -> ['observation.image', 'observation.wrist_image']
```

| | `lerobot/pi0_base` (expected) | `zhouqh/hrbench` (provided) |
|---|---|---|
| Camera 1 | `observation.images.base_0_rgb` | `observation.image` |
| Camera 2 | `observation.images.left_wrist_0_rgb` | `observation.wrist_image` |
| Camera 3 | `observation.images.right_wrist_0_rgb` | — (does not exist) |

Naming alone could be patched with `--rename_map` (e.g., map `observation.image
→ observation.images.base_0_rgb`, `observation.wrist_image →
observation.images.right_wrist_0_rgb`), but that still leaves
`observation.images.left_wrist_0_rgb` with no source. The model has weight
tensors sized for 3 image inputs — feeding only 2 is not just a config patch.

Pi0/pi05 LeRobotPolicy has a `--policy.empty_cameras=N` knob that the
`baseline/lerobot/src/lerobot/policies/` family supports (seen in our `policy`
config dump as `'empty_cameras': 0`), but it appears to add EMPTY camera slots,
not drop expected ones — we haven't confirmed it can satisfy the "missing 3rd
camera" case.

Same architecture mismatch applies to `lerobot/pi05_base`
(also a 3-camera Bridge-V2-style rig).

## Resolution (2026-05-23)

Both errors are unblocked by **switching the pretrained repo to `pi0_base` /
`pi05_base`** (fixes Error 1) and **passing `--rename_map` to map our 2
cameras onto 2 of the 3 expected slots** (fixes Error 2 — without code
changes, without rebuilding the dataset, without re-collecting data).

### Recipe

```bash
scripts/finetune.sh --model pi0 --slurm \
    --nodes 1 --gpus-per-node 8 --full \
    --time 3-00:00:00 --mem 0 \
    --exp-name pi0_base_ee \
    --pretrained lerobot/pi0_base \
    -- \
    --rename_map='{"observation.image":"observation.images.base_0_rgb","observation.wrist_image":"observation.images.right_wrist_0_rgb"}'
```

Swap `--model pi05 --pretrained lerobot/pi05_base` for pi05. Same
`--rename_map`. The `--` separator is required — `finetune_lerobot_v2.sh`
captures everything after it into `EXTRA` and splices it into the
`lerobot_train` argv unmodified
([`scripts/finetune_lerobot_v2.sh:63,163`](../scripts/finetune_lerobot_v2.sh)).

### Why it works (no architectural change required)

The doc's prior conclusion — *"feeding only 2 cameras is not just a config
patch"* — was incorrect. Three load-bearing facts in `lerobot v0.5.2` (all
shipping in `baseline/lerobot/src/`):

1. **The feature validator is skipped when `rename_map` is truthy** —
   `baseline/lerobot/src/lerobot/policies/factory.py:557`:
   ```python
   if not rename_map:
       validate_visual_features_consistency(cfg, features)
   ```
   Passing any non-empty `--rename_map` bypasses the `Feature mismatch`
   ValueError entirely. The validator is a *hint generator*, not a
   structural requirement.

2. **pi0 / pi05 / pi0_fast modeling has built-in handling for missing image
   keys** — same mechanism that powers the `empty_cameras` config knob.
   `baseline/lerobot/src/lerobot/policies/pi0/modeling_pi0.py:1173,1217`
   (mirrored in pi05 and pi0_fast):
   ```python
   present_img_keys = [key for key in self.config.image_features if key in batch]
   missing_img_keys = [key for key in self.config.image_features if key not in batch]
   ...
   # Create image features not present in the batch as fully 0 padded images
   for _num_empty_cameras in range(len(missing_img_keys)):
       img = torch.ones_like(img) * -1   # SigLIP-friendly padding
       mask = torch.zeros_like(mask)     # attention mask zeroed
       images.append(img); img_masks.append(mask)
   ```
   The missing camera's vision-tower slot is run on `-1`-padded input with
   attention mask = 0, so it contributes nothing to downstream layers. The
   pretrained weights for that slot become dead capacity but do not break
   anything.

3. **The normalize step silently no-ops missing keys** —
   `baseline/lerobot/src/lerobot/processor/normalize_processor.py:259,312`
   only normalizes features that are actually in the observation, and
   `_apply_transform` returns the tensor unchanged when stats for a key
   are absent. So pretrained stats sized for 3 cameras don't crash on our
   2-camera dataset.

### End-to-end data flow

| Stage | Effect |
|---|---|
| Dataloader yields | `{observation.image, observation.wrist_image, observation.state, action, ...}` |
| `RenameObservationsProcessorStep` | renames to `{observation.images.base_0_rgb, observation.images.right_wrist_0_rgb, ...}` |
| `NormalizerProcessorStep` | normalizes only the renamed image keys; silently no-ops the absent `left_wrist_0_rgb` |
| `make_policy` validator | **skipped** (rename_map present) |
| `pi0.prepare_images` | finds 2 present + 1 missing (`left_wrist_0_rgb`); appends `-1` placeholder and zero attention mask |
| SigLIP vision tower | processes 3 image tokens; the empty slot contributes nothing because attention is masked |

### Verification — sanity smokes

Both 200-step sanity runs on 1 × A100-80GB on the UMass cluster:

| | pi0 | pi05 |
|---|---|---|
| Pretrained | `lerobot/pi0_base` | `lerobot/pi05_base` |
| Job | 58275353 | 58275921 |
| Wall time to step 200 | 1m25s | 2m25s |
| Loss trajectory | 1.50 → 0.39 | 0.54 → 0.21 |
| Grad norm range | 4 – 36 | 3 – 18 |
| Checkpoint | ✅ saved (20 GB) | ✅ saved (21 GB) |
| Final state | `COMPLETED` | `COMPLETED` |
| Error 1 (DecodingError) | did not fire | did not fire |
| Error 2 (Feature mismatch) | did not fire | did not fire |

Both checkpoints are at
`runs/{pi0,pi05}/finetune/{pi0,pi05}_base_renamemap_sanity/checkpoints/last/pretrained_model/`
on the UMass cluster.

### Which wrist slot to map onto?

`pi0_base` / `pi05_base` were trained on a DROID-style 3-camera rig (base +
L wrist + R wrist). Our `observation.wrist_image` is a single
gripper-mounted view. Two options:

- **`observation.images.right_wrist_0_rgb`** *(recommended)* — most
  single-arm DROID/Bridge episodes use the right-wrist mount, so the
  pretrained right-wrist features are the closer initialization. Both
  smokes above used this mapping.
- **`observation.images.left_wrist_0_rgb`** — symmetric; no strong reason
  to prefer unless `right_wrist` underperforms after a real run.

The unused slot gets the `-1` placeholder fill described above.

### Caveats to watch in the full run

- **Normalization stats**: the pretrained checkpoint ships stats for 3
  image keys; our dataset provides stats for 2. The image we map to
  `base_0_rgb` will be normalized with the pretrained's `base_0_rgb`
  mean/std (DROID), not our genesis-hr-bench stats. Same for the wrist.
  Pi0 trains tens of K of steps so this typically washes out; if eval
  looks off, recompute stats post-finetune.
- **`empty_cameras=0`**: leave it alone. That knob *adds* empty cameras
  as additional expected inputs — the opposite of what we want.
- **Dead vision-tower capacity**: the `-1`-filled left-wrist slot is
  attention-masked to zero, so it's neither helpful nor harmful, just
  wasted parameters. If finetune eval underperforms vs from-scratch on
  certain tasks, the next experiment is duplicating `wrist_image` into
  both wrist slots (small `rename_map` extension or a custom processor
  step). Recommend running pi05_base full first and benchmarking before
  adding that complexity.

### Prompt for the FAIR agent

Copy-paste the following into the FAIR-side Claude session to run the
unblock end-to-end on FAIR Cloud (h200 partition):

> Pull `origin/meta/dev` and read the Resolution section in
> `docs/pi0-pi05-pretrained-load-issue.md`. Then verify the unblock on
> our cluster in two steps:
>
> 1. **Sanity smoke.** Submit a 1-GPU, 200-step sanity finetune of pi05
>    from `lerobot/pi05_base` with the rename_map recipe. Use
>    `scripts/finetune.sh --model pi05 --slurm --gpus-per-node 1
>    --sanity --time 1:00:00 --exp-name pi05_base_sanity_renamemap
>    --pretrained lerobot/pi05_base -- --rename_map='{"observation.image":"observation.images.base_0_rgb","observation.wrist_image":"observation.images.right_wrist_0_rgb"}'`.
>    Expected: training reaches step 200 with finite loss, checkpoint
>    saved, clean exit. UMass smokes hit step 200 in 1m25s (pi0) / 2m25s
>    (pi05) on a single A100; H200 should be faster. If either Error 1
>    or Error 2 fires, stop and report.
>
> 2. **Full run.** If the sanity smoke passes, kill any existing
>    from-scratch pi05 full job and relaunch with the same `--pretrained
>    lerobot/pi05_base` + `--rename_map` flags using
>    `scripts/finetune.sh --model pi05 --slurm --nodes 1 --gpus-per-node
>    8 --full --time 3-00:00:00 --mem 0 --exp-name pi05_base_full_ee
>    --pretrained lerobot/pi05_base -- --rename_map='...'`. Same recipe
>    for pi0 with `--model pi0 --pretrained lerobot/pi0_base`.
>
> Once both pretrained runs are off the ground, write a short progress
> note back in this doc comparing first-1k-step loss curves and wandb
> URLs against the from-scratch baselines at
> `multi-agent-world-model/{pi0,pi05}_genesis_hr_bench`. Do **not**
> commit the existing from-scratch wandb runs as the reference baseline
> for paper numbers — let the pretrained runs settle before deciding.
>
> Verification artifacts on UMass side (read-only reference): jobs
> `58275353` (pi0) and `58275921` (pi05), checkpoints under
> `runs/{pi0,pi05}/finetune/{pi0,pi05}_base_renamemap_sanity/checkpoints/last/`.

## Open questions for zhouqh

The blocker is purely on the pretrained side. To use pretrained pi0 / pi05 with
`zhouqh/hrbench`, we need one of these from zhouqh:

1. **A 2-camera pretrained checkpoint.** Does zhouqh have a pi0 / pi05 checkpoint
   trained on a 2-camera rig (the same head + single-wrist layout as
   `genesis_hr_bench_lerobot_v30`) that we should use instead of
   `lerobot/pi0_base` / `lerobot/pi05_base`?

2. **A `--rename_map` recipe + missing-camera strategy.** Has zhouqh fine-tuned
   pi0/pi05 from a 3-camera pretrained on this 2-camera data before? If yes,
   what was his `--rename_map`, and how did he handle the missing
   `left_wrist_0_rgb`? Common workarounds:

    - Duplicate `observation.wrist_image` into both `left_wrist_0_rgb` and
      `right_wrist_0_rgb` slots.
    - Feed a zero/black image for `left_wrist_0_rgb`.
    - Use `--policy.empty_cameras` to drop one camera from the model graph at
      finetune time (needs verification in our `lerobot v0.5.2`).

3. **A different pretrained-load entry point.** Is there a different lerobot
   command (e.g., `--policy.pretrained_path=...` instead of
   `--policy.path=...`, or a manual state_dict load step) that accepts partial
   weight loading and re-initializes the missing-camera weights?

The questions above are now optional — the rename_map unblock above is
sufficient to load `pi0_base` / `pi05_base` against the 2-camera dataset.
The from-scratch full runs (wandb projects
`multi-agent-world-model/pi0_genesis_hr_bench` and
`multi-agent-world-model/pi05_genesis_hr_bench`) remain healthy and can
serve as the comparison baseline against any pretrained run launched with
the recipe in [Resolution](#resolution-2026-05-23).

## Related files

- Wrapper that translates `--pretrained X` → `--policy.path=X`:
  [`scripts/finetune_lerobot_v2.sh:62,112`](../scripts/finetune_lerobot_v2.sh).
- Top-level dispatcher: [`scripts/finetune.sh`](../scripts/finetune.sh).
- Feature consistency check that raises Error 2:
  `baseline/lerobot/src/lerobot/policies/utils.py:213`.
- Pipeline doc that prescribes lerobot for pi0/pi05:
  [`docs/vla-finetune-pipeline.md`](./vla-finetune-pipeline.md).
