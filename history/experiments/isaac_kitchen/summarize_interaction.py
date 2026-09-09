"""Collect every condition, including failures, from the C1 simulator experiment."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for name in ('legacy_normal', 'c1_normal', 'c1_delay', 'c1_pause'):
    folder = root / name
    if not (folder / 'result.json').exists():
        rows.append(dict(condition=name, status='missing_result'))
        continue
    result = json.loads((folder / 'result.json').read_text())
    interaction = json.loads((folder / 'interaction_result.json').read_text())
    handover = result.get('handover') or {}
    events = {e['to']: e['step'] for e in interaction['events']}
    wait_rows = [r for r in interaction['trace'] if r['state'] == 'wait']
    rows.append(dict(condition=name, status=result['status'],
                     wait_s=(events['receive'] - events['wait']) / 60 if 'receive' in events else None,
                     max_wait_palm_error_m=max((r['palm_error_m'] for r in wait_rows), default=None),
                     capture_step=handover.get('capture_step'),
                     release_step=handover.get('release_step'),
                     carry_step=handover.get('human_carry_step'),
                     final_hold_s=handover.get('final_hold_s'),
                     max_follow_error_m=handover.get('max_follow_error_since_capture_m'),
                     carry_distance_m=handover.get('human_carry_distance_m'),
                     perturbations=interaction['perturbations'], events=interaction['events']))
summary = dict(scope='C1-inspired behavior adaptation, existing IK and spring proxy; not an upstream reproduction',
               conditions=rows)
(root / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
