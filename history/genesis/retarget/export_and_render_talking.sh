#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH -c 16
#SBATCH -o /scratch4/workspace/qinhongzhou_umass_edu-simple/code/mawm/genesis-hr-bench/output/talking_export.log
#SBATCH -e /scratch4/workspace/qinhongzhou_umass_edu-simple/code/mawm/genesis-hr-bench/output/talking_export.log
#SBATCH -J talking_export

set -e

BLENDER="/work/pi_chuangg_umass_edu/qinhongzhou/blender/blender-4.2.1-linux-x64/blender"
RETARGET_DIR="/scratch4/workspace/qinhongzhou_umass_edu-simple/code/mawm/genesis-hr-bench/retarget"
ROBOTWIN_DIR="/scratch4/workspace/qinhongzhou_umass_edu-simple/code/mawm/RoboTwin_genesis"
PKL="$RETARGET_DIR/output_motion/motion.pkl"
OUTDIR="/scratch4/workspace/qinhongzhou_umass_edu-simple/code/mawm/genesis-hr-bench/output"

PYTHON=/work/pi_chuangg_umass_edu/qinhongzhou/conda/robotwin/bin/python
export PATH="/work/pi_chuangg_umass_edu/qinhongzhou/conda/robotwin/bin:$PATH"
export PYOPENGL_PLATFORM=egl
export MESA_GL_VERSION_OVERRIDE=3.3

echo "=== Step 1: Export already done, rendering Deliver motion ==="
cd "$ROBOTWIN_DIR"
$PYTHON script/play_motion_standalone.py --list --motion-pkl "$PKL"

echo ""
echo "=== Step 2: Render Deliver motion with frame numbers ==="
$PYTHON script/play_motion_standalone.py "Deliver" \
    --motion-pkl "$PKL" \
    -o "$OUTDIR/Deliver_frames.mp4"

echo ""
echo "=== Done ==="
