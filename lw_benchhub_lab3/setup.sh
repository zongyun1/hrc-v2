#!/usr/bin/env bash
# Run on aicr after copying this directory to the desired independent root.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
SOURCE_BASE=${LW_SOURCE_BASE:-/scratch/jiabenchen_umass/yz/lw-benchhub-check/source}
RUNTIME=${LW_RUNTIME:-/scratch/jiabenchen_umass/moment-contact-memory/external/isaaclab300b2p1_ngc_container_v1/isaac-lab_3.0.0-beta2-post1_amd64.sif}
if [[ ! -d "$ROOT/source" ]]; then
    cp -a "$SOURCE_BASE" "$ROOT/source"
fi
test "$(git -C "$ROOT/source" rev-parse HEAD)" = b2bcb2d00edef691f9fcc49039cbf0bcc7464605
test "$(git -C "$ROOT/source/third_party/IsaacLab-Arena" rev-parse HEAD)" = c7b70779f103e10d690d1a13863e8d77da7fc782
python3 "$ROOT/migrate.py" "$ROOT/source"
mkdir -p "$ROOT/deps" "$ROOT/home" "$ROOT/cache" "$ROOT/logs"
apptainer exec --bind /scratch:/scratch --home "$ROOT/home:/root" "$RUNTIME" \
    /isaac-sim/python.sh -m pip install --no-deps --target "$ROOT/deps" \
    -r "$ROOT/requirements-extra.txt"
apptainer exec --bind /scratch:/scratch --home "$ROOT/home:/root" "$RUNTIME" \
    /isaac-sim/python.sh -m pip install --no-deps --target "$ROOT/deps" \
    -e "$ROOT/source" -e "$ROOT/source/third_party/IsaacLab-Arena"
