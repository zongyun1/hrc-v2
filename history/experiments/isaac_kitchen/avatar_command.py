"""Write a command for --avatar_command_file (no Isaac/Genesis dependency)."""
import argparse
import json
import os
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--file', required=True)
sub = parser.add_subparsers(dest='action', required=True)
reach = sub.add_parser('reach')
reach.add_argument('--hand', choices=['left', 'right'], default='right')
reach.add_argument('--target', nargs=3, type=float, required=True)
reach.add_argument('--duration', type=float, default=2.0)
base = sub.add_parser('base')
base.add_argument('--position', nargs=3, type=float, required=True)
base.add_argument('--yaw', type=float, required=True, help='world yaw in radians')
sub.add_parser('reset')
args = vars(parser.parse_args())
path = Path(args.pop('file')).resolve()
path.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
    json.dump(args, f)
os.replace(f.name, path)
print(json.dumps(args))
