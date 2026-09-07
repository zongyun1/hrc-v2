#!/bin/bash
#SBATCH -J nyx_envmap_probe
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=sm_70
#SBATCH --exclude=umd-cscdr-gpu[001-002],uri-gpu[001-008,015-016],gpu[012,022-024,051],ials-gpu036
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -t 0:45:00
#SBATCH -o job_logs/nyx_envmap_probe_%j.log
#SBATCH -e job_logs/nyx_envmap_probe_%j.log

set -x
cd /scratch4/workspace/qinhongzhou_umass_edu-simple/code/mawm/genesis-hr-bench
PY=/scratch4/workspace/qinhongzhou_umass_edu-simple/yz/env/MAWM_nyx_clean/bin/python
OUT=data/background/nyx_envmap_probe
EXR=$PWD/assets/exr

export GENESIS_BACKEND=gpu

# One scene build per process (NYX limitation) -> one python call per config.
$PY scripts/probe_nyx_envmap.py $OUT/0_baseline_no_envmap.png
$PY scripts/probe_nyx_envmap.py $OUT/1_brown_photostudio.png   $EXR/brown_photostudio_02_4k.exr
$PY scripts/probe_nyx_envmap.py $OUT/2_glasshouse_interior.png $EXR/glasshouse_interior_4k.exr
$PY scripts/probe_nyx_envmap.py $OUT/3_university_workshop.png $EXR/university_workshop_4k.exr
$PY scripts/probe_nyx_envmap.py $OUT/4_brown_rot90.png         $EXR/brown_photostudio_02_4k.exr 90
$PY scripts/probe_nyx_envmap.py $OUT/5_brown_mult2.png         $EXR/brown_photostudio_02_4k.exr 0 2.0

echo DONE
