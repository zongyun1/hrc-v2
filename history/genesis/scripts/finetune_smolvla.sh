#!/bin/bash
# Back-compat shim: SmolVLA finetuning now goes through the generic lerobot
# v0.5.2 VLA driver, scripts/finetune_lerobot_v2.sh, which also serves
# pi0 / pi05 / pi0_fast and supports the NNODES x GPUS_PER_NODE contract.
#
# Forwards everything with --policy-type smolvla prepended.
exec "$(dirname "${BASH_SOURCE[0]}")/finetune_lerobot_v2.sh" --policy-type smolvla "$@"
