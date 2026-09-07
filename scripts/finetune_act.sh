#!/bin/bash
# Back-compat shim: ACT finetuning now goes through scripts/finetune_lerobot.sh.
# Forwards everything except --policy-type to the generic driver with
# --policy-type=act prepended.
exec "$(dirname "${BASH_SOURCE[0]}")/finetune_lerobot.sh" --policy-type act "$@"
