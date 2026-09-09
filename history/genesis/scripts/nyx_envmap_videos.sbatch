#!/bin/bash
#SBATCH -J nyx_envmap_videos
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=sm_70
#SBATCH --exclude=umd-cscdr-gpu[001-002],uri-gpu[001-008,015-016],gpu[012,022-024,051],ials-gpu036
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH -t 6:00:00
#SBATCH -o job_logs/nyx_envmap_videos_%j.log
#SBATCH -e job_logs/nyx_envmap_videos_%j.log

# Review videos: real tasks rendered under NYX with EXR HDRI backgrounds
# (env-map wiring in envs/camera.py). One collect.py process per episode —
# NYX allows a single scene build per process.

set -x
cd /scratch4/workspace/qinhongzhou_umass_edu-simple/code/mawm/genesis-hr-bench
PY=/scratch4/workspace/qinhongzhou_umass_edu-simple/yz/env/MAWM_nyx_clean/bin/python
OUT=data/background/nyx_envmap_videos
mkdir -p "$OUT/configs"

export GENESIS_BACKEND=gpu

run_one () {  # run_one <task> <exr-basename-or-none> <run-name>
    local task=$1 exr=$2 name=$3
    local cfg=$OUT/configs/${name}.yml
    {
        echo "renderer: nyx"
        echo "nyx:"
        echo "  spp: 4"
        echo "  denoise: true"
        echo "  open_window: false"
        if [ "$exr" != "none" ]; then
            echo "  env_texture: assets/exr/${exr}.exr"
        fi
    } > "$cfg"
    $PY scripts/collect.py --task "$task" --config "$cfg" \
        --episodes 1 --save-dir "$OUT/$name" || echo "RUN_FAILED $name"
    # Flatten: user reviews a single folder of <run-name>.mp4, no subfolders.
    if [ -f "$OUT/$name/video/seed_0.mp4" ]; then
        mv "$OUT/$name/video/seed_0.mp4" "$OUT/$name.mp4"
    fi
    rm -rf "$OUT/$name"
}

# Baseline (no env map, floor visible) exists from the previous round; these
# 6 re-render with the HDRI background AND ground visuals hidden.
run_one pour_water        brown_photostudio_02_4k  pour_water__brown_photostudio
run_one pour_water        glasshouse_interior_4k   pour_water__glasshouse
run_one dump_bin_neutral  university_workshop_4k   dump_bin_neutral__university_workshop
run_one dump_bin_neutral  brown_photostudio_02_4k  dump_bin_neutral__brown_photostudio
run_one open_microwave    glasshouse_interior_4k   open_microwave__glasshouse
run_one open_microwave    church_meeting_room_4k   open_microwave__church_meeting_room

rm -rf "$OUT/configs"
echo ALL_DONE
find "$OUT" -name "*.mp4" | sort
