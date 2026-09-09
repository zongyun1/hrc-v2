#!/usr/bin/env bash
#SBATCH --job-name=serve-tea-policy
#SBATCH --partition=rtx-devel,rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/jiabenchen_umass/yz/job_logs/serve_tea_policy_%j.log
set -euo pipefail
cd /scratch/jiabenchen_umass/yz/hrc-v2
export ROBOCASA_PROBE_MODE=robot
export ROBOCASA_ROBOT_STEPS="${ROBOCASA_ROBOT_STEPS:-10800}"
export ROBOCASA_HOLD_STEPS="${ROBOCASA_HOLD_STEPS:-240}"
export ROBOCASA_SERVE_TEA_CONTACT_PROBE=0
bash robocasa_migration/run_probe.sbatch ServeTea
python3 - "outputs/robocasa_migration/probes/${SLURM_JOB_ID}/ServeTea/result.json" <<'PY'
import json
import sys
result = json.load(open(sys.argv[1]))
assert result["mode"] == "original_pandaomron_serve_tea", result["mode"]
assert result["robot_task_success"], result["trace"][-1]["robot"]
print("SERVE_TEA_POLICY_SUCCESS", sys.argv[1])
PY
