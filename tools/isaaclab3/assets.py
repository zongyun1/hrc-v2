"""Verify/copy upstream LFS payloads from a separately downloaded source checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Upstream checkout with LFS payloads downloaded')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    assets = json.loads((args.root/'upstream.json').read_text())['lfs_assets']
    missing = []
    for item in assets:
        target = args.root/item['path']
        candidate = target if target.exists() else (args.source/item['path'] if args.source else target)
        if not candidate.is_file() or candidate.stat().st_size != item['size'] or digest(candidate) != item['sha256']:
            missing.append(item['path'])
            continue
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, target)
    if missing:
        parser.exit(1, 'Missing or mismatched upstream assets:\n'+'\n'.join(missing)+'\n')
    print(f'Verified {len(assets)} upstream LFS assets in {args.root.resolve()}')


if __name__ == '__main__':
    main()
