#!/usr/bin/env bash
#SBATCH --job-name=robocasa-convert
#SBATCH --partition=rtx-devel,rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/jiabenchen_umass/yz/job_logs/robocasa_convert_%j.log
set -euo pipefail
ROOT=/scratch/jiabenchen_umass/yz
PROJECT="$ROOT/hrc-v2"
RUNTIME=/scratch/jiabenchen_umass/moment-contact-memory/external/isaaclab300b2p1_ngc_container_v1/isaac-lab_3.0.0-beta2-post1_amd64.sif
mkdir -p "$ROOT/.cache/robocasa-kit-home" "$ROOT/.cache/robocasa-kit-cache" "$ROOT/.cache/robocasa-kit-logs"
export APPTAINERENV_ACCEPT_EULA=Y APPTAINERENV_OMNI_KIT_ACCEPT_EULA=YES APPTAINERENV_PRIVACY_CONSENT=Y
export APPTAINERENV_ISAAC_PATH=/isaac-sim APPTAINERENV_OMNI_KIT_ALLOW_ROOT=1
export PYTHONUNBUFFERED=1
unset CONDA_PREFIX CONDA_DEFAULT_ENV
cd "$PROJECT"
tasks=("${@}")
if [ ${#tasks[@]} -eq 0 ]; then tasks=(PickPlaceCounterToSink OpenCabinet OpenMicrowave); fi
failed=0
for task in "${tasks[@]}"; do
    apptainer exec --nv --fakeroot --writable-tmpfs --bind /scratch:/scratch \
        --home "$ROOT/.cache/robocasa-kit-home:/root" \
        --bind "$ROOT/.cache/robocasa-kit-cache:/isaac-sim/kit/cache" \
        --bind "$ROOT/.cache/robocasa-kit-logs:/isaac-sim/kit/logs" \
        "$RUNTIME" /isaac-sim/python.sh robocasa_migration/convert_scene.py \
        --headless --task "$task" \
        --source "${ROBOCASA_MIGRATION_ROOT:-outputs/robocasa_migration}/source" \
        --output "${ROBOCASA_MIGRATION_ROOT:-outputs/robocasa_migration}/usd" || { failed=1; continue; }
    if [ "${ROBOCASA_REPAIR_MANIPULATION_MESHES:-0}" = 1 ]; then
        python3 - "${ROBOCASA_MIGRATION_ROOT:-outputs/robocasa_migration}" "$task" <<'PY' || failed=1
import json
import subprocess
import sys
from pathlib import Path
root, task = Path(sys.argv[1]), sys.argv[2]
assert task == 'ServeTea', 'Manipulation repair currently supports ServeTea only'
report_path = root / 'usd' / task / 'conversion.json'
report = json.loads(report_path.read_text())
assert report['status'] == 'converted', report
repair_path = report_path.parent / 'mesh_repair.json'
check_path = report_path.parent / 'mesh_after_repair.json'
command = ['.venv-robocasa/bin/python', 'robocasa_migration/inspect_collision_meshes.py',
           '--source', str(root / 'source' / task / 'scene.xml'), '--usd', report['usd_path']]
subprocess.run([*command, '--output', str(repair_path), '--repair'], check=True)
subprocess.run([*command, '--output', str(check_path)], check=True)
checks = json.loads(check_path.read_text())
assert checks and max(r['vertex_error_m'] for r in checks) < 1e-6, checks
subprocess.run(['.venv-robocasa/bin/python', 'robocasa_migration/inspect_object_dynamics.py',
                '--root', str(root), '--repair'], check=True)
report['mesh_frame_repair'] = {'repair_report': str(repair_path), 'check_report': str(check_path),
    'max_vertex_error_m': max(r['vertex_error_m'] for r in checks)}
report['source_object_inertia_restored'] = True
report_path.write_text(json.dumps(report, indent=2))
PY
    fi
done
python3 - "${tasks[@]}" <<'PY'
import json
import sys
import os
from pathlib import Path
root = Path(os.environ.get('ROBOCASA_MIGRATION_ROOT', 'outputs/robocasa_migration')) / 'usd'
for task in sys.argv[1:]:
    result = json.loads((root / task / 'conversion.json').read_text())
    assert result['status'] == 'converted', result
PY
exit "$failed"
