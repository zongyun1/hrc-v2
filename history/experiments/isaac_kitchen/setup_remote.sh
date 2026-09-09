#!/usr/bin/env bash
set -euo pipefail
ROOT=/scratch/jiabenchen_umass/yz
RUNTIME=/scratch/jiabenchen_umass/moment-contact-memory/external/isaaclab300b2p1_ngc_container_v1
ASSETS="$ROOT/assets/lightwheel-kitchen"
test -f "$RUNTIME/isaac-lab_3.0.0-beta2-post1_amd64.sif"
mkdir -p "$ASSETS" "$ROOT/job_logs"
if [ ! -f "$ASSETS/Collected_KitchenRoom/KitchenRoom.usd" ]; then
    curl -fL --retry 3 https://media.githubusercontent.com/media/LightwheelAI/Lightwheel_Kitchen/main/Lightwheel_Kitchen.zip -o "$ASSETS/kitchen.zip"
    echo "721fb9e47d09b2c43a7db396c1d63c659e987d5925db0a5e78cf4fd40cd0b0b0  $ASSETS/kitchen.zip" | sha256sum -c -
    python3 -m zipfile -e "$ASSETS/kitchen.zip" "$ASSETS"
fi
AVATAR="$ROOT/assets/avatars/custom_Adrian_Keller.glb"
if [ ! -f "$AVATAR" ]; then
    AVATAR_SOURCE="${AVATAR_SOURCE:-$ROOT/genesis-hrc/asset/avatars/models/custom_Adrian_Keller.glb}"
    test -f "$AVATAR_SOURCE"
    mkdir -p "$ROOT/assets/avatars"
    cp "$AVATAR_SOURCE" "$AVATAR"
fi
echo "Ready: Isaac Lab container + kitchen + legacy human avatar"
