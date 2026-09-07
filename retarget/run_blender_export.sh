#!/bin/bash
# Export motion from Blender to motion.pkl
# Usage: ./run_blender_export.sh [blend_file]

BLENDER="/work/pi_chuangg_umass_edu/qinhongzhou/blender/blender-4.2.1-linux-x64/blender"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLEND="${1:-$SCRIPT_DIR/test.blend}"

# if [[ ! -f "$BLEND" ]]; then
#     echo "Error: Blend file not found: $BLEND"
#     exit 1
# fi

echo "Blender: $BLENDER"
echo "Blend:   $BLEND"
echo "Output:  $SCRIPT_DIR/motion.pkl"
echo ""

# "$BLENDER" "$BLEND" -b -P "$SCRIPT_DIR/export_motion_from_blender.py" -- \
    # --output "$SCRIPT_DIR/motion.pkl" --merge "$SCRIPT_DIR/motion.pkl"

# Playback / verify exported motion:
#   python script/play_motion_standalone.py --list --motion-pkl $SCRIPT_DIR/motion.pkl
#   python script/play_motion_standalone.py open_door --motion-pkl $SCRIPT_DIR/motion.pkl -o output.mp4

# python script/cut_motion_pkl.py $SCRIPT_DIR/motion.pkl open_door --part1 3 96 --name1 open_and_hold_the_door --target-pkl text-to-motion/motion.pkl

python script/play_motion_standalone.py neutral_phone_pass_1 --motion-pkl text-to-motion/motion.pkl -o output.mp4

# deliver_to_human_safety_1 neutral_phone_pass_1

# python script/play_motion_standalone.py --list --motion-pkl text-to-motion/motion.pkl

# Extract hand position at frame N, draw red sphere at that position, save debug image:
#   python script/play_motion_standalone.py <motion_name> --motion-pkl $SCRIPT_DIR/motion.pkl --extract --frame N --hand left|right --debug-sphere -o hand_debug.png
# python script/play_motion_standalone.py neutral_phone_pass_1 --motion-pkl $SCRIPT_DIR/motion.pkl --extract --frame 33 --hand right 
# --debug-sphere -o hand_debug.png

# Cut motion into segments (run from repo root):
#   python script/cut_motion_pkl.py text-to-motion/motion.pkl neutral_phone_pass_1 --part1 50 84 --part2 84 94 --name1 deliver_to_human_safety_1 --name2 deliver_to_human_safety_2
