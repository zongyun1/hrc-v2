#!/usr/bin/env bash
# Install this root checkout inside the existing Isaac Lab 3 container.
set -euo pipefail
PROJECT=$(cd "$(dirname "$0")/../.." && pwd)
STATE="$PROJECT/external/isaaclab3"
RUNTIME=${LW_RUNTIME:-/scratch/jiabenchen_umass/moment-contact-memory/external/isaaclab300b2p1_ngc_container_v1/isaac-lab_3.0.0-beta2-post1_amd64.sif}
mkdir -p "$STATE/deps" "$STATE/home" "$STATE/cache" "$STATE/logs"
apptainer exec --bind /scratch:/scratch --bind "$PROJECT:$PROJECT" --home "$STATE/home:/root" "$RUNTIME" \
    /isaac-sim/python.sh -m pip install --no-deps --target "$STATE/deps" \
    -r "$PROJECT/tools/isaaclab3/requirements-extra.txt"
apptainer exec --bind /scratch:/scratch --bind "$PROJECT:$PROJECT" --home "$STATE/home:/root" "$RUNTIME" \
    /isaac-sim/python.sh -m pip install --no-deps --upgrade --target "$STATE/deps" \
    -e "$PROJECT" -e "$PROJECT/third_party/IsaacLab-Arena"
