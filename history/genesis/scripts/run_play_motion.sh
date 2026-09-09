#!/bin/bash
#SBATCH -p gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -t 00:15:00
#SBATCH -o scripts/run_play_motion.log
#SBATCH -e scripts/run_play_motion.log
#SBATCH -J play_motion

set -e
cd /scratch4/workspace/qinhongzhou_umass_edu-simple/yz/code/genesis-hr-bench

export PYOPENGL_PLATFORM=egl
export MESA_GL_VERSION_OVERRIDE=3.3

PYTHON=/work/pi_chuangg_umass_edu/qinhongzhou/conda/robotwin/bin/python
MOTION="${MOTION:-Shove_Reaction}"
OUTPUT="${OUTPUT:-shove_reaction.mp4}"

echo "Rendering motion: $MOTION -> $OUTPUT"
$PYTHON scripts/play_motion.py --motion "$MOTION" -o "$OUTPUT"
echo "Done."
