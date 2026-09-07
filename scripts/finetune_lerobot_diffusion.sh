#!/bin/bash
# Shim: forwards to scripts/finetune_lerobot.sh with --policy-type=diffusion.
# Named *_lerobot_diffusion* to disambiguate from finetune_diffusion_policy.sh,
# which trains Chi et al.'s standalone Diffusion Policy implementation.
exec "$(dirname "${BASH_SOURCE[0]}")/finetune_lerobot.sh" --policy-type diffusion "$@"
