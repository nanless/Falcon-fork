"""List actual directories and a bounded sample of each mounted raw directory."""
from __future__ import annotations
import argparse
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from snapshot_catalog import write_new_json

DEFAULT = Path('/aispeech-cfs/voiceprint/data/multimodal')
SENSITIVE = re.compile(r'(?i)(?:token|secret|credential|password|signature|gh[pousr]_|hf_)')


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def inspect(path, limit):
    result = {'path': str(path), 'entries': [], 'truncated': False, 'error': None, 'omitted_names': 0}
    try:
        with os.scandir(path) as entries:
            for index, entry in enumerate(entries):
                if index >= limit:
                    result['truncated'] = True
                    break
                if SENSITIVE.search(entry.name):
                    result['omitted_names'] += 1
                    continue
                try:
                    stat = entry.stat(follow_symlinks=False)
                    is_file = entry.is_file(follow_symlinks=False)
                    result['entries'].append({'name': entry.name,
                        'kind': 'dir' if entry.is_dir(follow_symlinks=False) else 'file' if is_file else 'other',
                        'bytes': stat.st_size if is_file else None})
                except OSError as exc:
                    result['entries'].append({'name': entry.name, 'error': type(exc).__name__})
    except OSError as exc:
        result['error'] = type(exc).__name__
    return result


def scan(root, limit):
    start = now(); items = []; omitted_root_names = 0
    for path in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if SENSITIVE.search(path.name):
            omitted_root_names += 1
            continue
        if path.is_symlink():
            items.append({'name': path.name, 'kind': 'symlink'})
        elif path.is_dir():
            raw = path / 'raw'
            items.append({'name': path.name, 'kind': 'directory', 'top_level': inspect(path, limit),
                          'raw_level': inspect(raw, limit) if raw.is_dir() else None})
        elif path.is_file():
            items.append({'name': path.name, 'kind': 'file', 'bytes': path.stat().st_size})
        else:
            items.append({'name': path.name, 'kind': 'other'})
    return {'start': start, 'end': now(), 'cold_root': str(root), 'limit_per_directory': limit, 'omitted_root_names': omitted_root_names,
            'scope': 'Direct listing/stat of root, each top-level directory and raw directory; bounded, non-recursive, without decoding. Truncated entry counts are lower bounds.',
            'directories': items}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=DEFAULT)
    p.add_argument('--limit', type=int, default=50)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.limit <= 1000:
        p.error('limit must be between 1 and 1000')
    result = scan(args.root, args.limit)
    write_new_json(args.output, result)
    print(f"listed {len(result['directories'])} top-level entries in {args.root}; saved {args.output}")


if __name__ == '__main__':
    main()
