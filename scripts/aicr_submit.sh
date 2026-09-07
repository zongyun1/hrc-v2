#!/usr/bin/env bash
set -euo pipefail

REMOTE="aicr:/scratch/jiabenchen_umass/yz/code"
REMOTE_DIR="/scratch/jiabenchen_umass/yz/code"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <slurm-script> [sbatch args...]" >&2
  exit 2
fi

SCRIPT="$1"; shift
[[ -f "$SCRIPT" ]] || { echo "Missing script: $SCRIPT" >&2; exit 2; }

rsync -az --delete --exclude '.git/' --exclude '__pycache__/' ./ "$REMOTE/"
REMOTE_SCRIPT="${SCRIPT#./}"
ssh aicr "cd '$REMOTE_DIR' && sbatch '$REMOTE_SCRIPT'$(printf ' %q' \"$@\")"
