#!/usr/bin/env bash
# Run inside the Ubuntu container, after installing Isaac Sim 5.0.
set -euo pipefail
CHECK_ROOT=/scratch/jiabenchen_umass/yz/lw-benchhub-check
cd "$CHECK_ROOT"
UV=/home/jiabenchen_umass/.local/bin/uv
P="$CHECK_ROOT/runtime/bin/python"
export UV_LINK_MODE=copy
"$UV" pip install --python "$P" --python-platform x86_64-manylinux_2_35 \
    'setuptools<82' wheel toml 'numpy<2' 'torch==2.7.0+cu128' 'torchvision==0.22.0+cu128' \
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
# Same flatdict workaround documented by upstream install.sh; restore source after packaging.
LAB="$CHECK_ROOT/source/third_party/IsaacLab-Arena/submodules/IsaacLab"
sed -i 's/flatdict==4.0.1/flatdict==4.0.0/' "$LAB/source/isaaclab/setup.py"
trap 'git -C "$LAB" restore source/isaaclab/setup.py' EXIT
"$UV" pip install --python "$P" --python-platform x86_64-manylinux_2_35 --no-build-isolation \
    -e "$LAB/source/isaaclab" -e "$LAB/source/isaaclab_assets" \
    -e "$LAB/source/isaaclab_tasks" -e "$LAB/source/isaaclab_rl" \
    -e "$CHECK_ROOT/source/third_party/IsaacLab-Arena" -e "$CHECK_ROOT/source" \
    'torch==2.7.0' 'torchvision==0.22.0' 'lightwheel-sdk==1.0.3' 'warp-lang==1.7.1' \
    imageio imageio-ffmpeg scipy h5py omegaconf hydra-core mujoco
"$UV" pip freeze --python "$P" > "$CHECK_ROOT/runtime-freeze.txt"
