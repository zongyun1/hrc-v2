#!/bin/bash
# Shim: forwards to scripts/finetune_lerobot.sh with --policy-type=vqbet.
#
# KNOWN ISSUE (2026-05-12): VQ-BeT's validate_features() requires exactly
# one image input, but our LeRobot dataset (genesis_hr_bench_lerobot)
# ships head + wrist (2 cameras). lerobot/factory.py:make_policy
# unconditionally rebuilds cfg.input_features from the dataset metadata,
# so passing --policy.input_features on the CLI does *not* drop the wrist
# camera. You will hit:
#   ValueError: You must provide only one image among the inputs.
# in lerobot/policies/vqbet/configuration_vqbet.py.
#
# TODO: extend scripts/convert_genesis_hr_bench_to_lerobot.py with a
# --single-image flag that omits observation.wrist_image and writes a
# sibling dataset under repo_id genesis-hr-bench/genesis_hr_bench_lerobot_singlecam,
# then default this script to that repo_id. Until that lands, VQ-BeT is
# wired up to the server / client / run_vla_eval.sh dispatch but you
# cannot finetune against the 2-cam dataset.

set -e
DATASET_REPO_ID="${DATASET_REPO_ID:-genesis-hr-bench/genesis_hr_bench_lerobot_singlecam}"
SINGLECAM_DIR="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}/genesis-hr-bench/genesis_hr_bench_lerobot_singlecam"
if [[ ! -d "$SINGLECAM_DIR" ]]; then
  cat <<EOF >&2
[finetune_vqbet] ERROR: single-image dataset not found at
    $SINGLECAM_DIR
VQ-BeT requires a 1-camera variant of the genesis_hr_bench LeRobot dataset
(see the comment block at the top of this script for the full story).

Workarounds:
  1. Build the single-cam variant once a --single-image flag is added to
     scripts/convert_genesis_hr_bench_to_lerobot.py.
  2. Or set DATASET_REPO_ID=<your existing 1-cam repo_id> in the env.
EOF
  exit 3
fi

export DATASET_REPO_ID
exec "$(dirname "${BASH_SOURCE[0]}")/finetune_lerobot.sh" --policy-type vqbet "$@"
