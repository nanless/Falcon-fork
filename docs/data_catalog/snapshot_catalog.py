"""Read the live multimodal registry, status files and cold receipts into JSON."""
from __future__ import annotations
import argparse
import json
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

HOT = Path('/root/group-shared/voiceprint/data/multimodal')
COLD = Path('/aispeech-cfs/voiceprint/data/multimodal')
STATUS_FIELDS = ('state', 'updated_at', 'updated_at_utc', 'expected_files', 'completed_files',
                 'verified_files', 'expected_bytes', 'completed_bytes', 'verified_bytes')
RECEIPT_FIELDS = ('slug', 'action', 'mode', 'file_count', 'total_bytes', 'archived_at',
                  'updated_at', 'cold_payload', 'cold_verified', 'post_cleanup_cold_verified')
SENSITIVE = re.compile(r'(?i)(?:password|passwd|secret|credential|authorization\s*[:=]|bearer\s+|api[_-]?key|access[_-]?key|signature|x-amz-|x-tos-|private[_ -]?key|gh[pousr]_[a-z0-9]{20,}|hf_[a-z0-9]{20,}|AKIA[A-Z0-9]{16})')
SAFE_SLUG = re.compile(r'^[a-z0-9][a-z0-9_-]*$')


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def safe_string(value):
    if not isinstance(value, str):
        return None
    # Free-form metadata may embed signed URLs; keep only explicitly safe source URLs.
    decoded = unquote(unquote(value))
    if SENSITIVE.search(decoded) or '://' in decoded or ('?' in decoded and '=' in decoded):
        return '[已省略可能含敏感信息的字段]'
    return value


def scalar_fields(data, names):
    out = {}
    for key in names:
        value = data.get(key)
        if isinstance(value, str): out[key] = safe_string(value)
        elif isinstance(value, (int, float, bool)): out[key] = value
    return out


def clean_source(value):
    if not isinstance(value, str):
        return None
    if value.startswith(('http://', 'https://')):
        parts = urlsplit(value)
        host = parts.netloc.rsplit('@', 1)[-1]  # remove userinfo as well as query/fragment
        if not host or SENSITIVE.search(unquote(unquote(host + parts.path))):
            return '[已省略可能含敏感信息的来源]'
        return urlunsplit((parts.scheme, host, parts.path, '', ''))
    # Unstructured source strings can contain credentials without URL delimiters.
    return None


def extras(root, known):
    return [{'name': safe_string(p.name), 'path': safe_string(str(p)),
             'kind': 'directory' if p.is_dir() else 'file', 'symlink': p.is_symlink()}
            for p in sorted(root.iterdir(), key=lambda p: p.name.lower()) if p.name not in known]


def snapshot(hot, cold):
    captured = now()
    registry = hot / '_catalog' / 'dataset_registry.json'
    entries = json.loads(registry.read_text())['datasets']
    names = [item['slug'] for item in entries]
    if any(not isinstance(s, str) or (not SAFE_SLUG.fullmatch(s) or SENSITIVE.search(s)) for s in names):
        raise ValueError('invalid dataset slug in registry')
    if len(names) != len(set(names)):
        raise ValueError('duplicate dataset slug in registry')
    rows = []
    for item in entries:
        slug = item['slug']; h, c = hot / slug, cold / slug
        status_path = h / 'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        receipts = []
        for name in ('archive_receipt.json', 'incremental_receipt.json'):
            path = c / name
            if path.exists():
                record = json.loads(path.read_text())
                receipts.append({'path': safe_string(str(path)), **scalar_fields(record, RECEIPT_FIELDS)})
        rows.append({'slug': slug, 'name': safe_string(item.get('name')),
                     'description': safe_string(item.get('description')),
                     'source': clean_source(item.get('source')), 'license': safe_string(item.get('license')),
                     'initial_state': safe_string(item.get('initial_state')),
                     'cold_reference': safe_string(item.get('cold_reference')),
                     'hot_directory': safe_string(str(h)), 'hot_directory_exists': h.is_dir(),
                     'status_file': safe_string(str(status_path)) if status_path.exists() else None,
                     'status_snapshot': scalar_fields(status, STATUS_FIELDS),
                     'cold_directory': safe_string(str(c)), 'cold_directory_exists': c.is_dir(),
                     'cold_raw_directory_exists': (c / 'raw').is_dir(), 'historical_receipts': receipts})
    known = set(names)
    hot_extras = extras(hot, known); hot_extra_captured_at = now()
    cold_extras = extras(cold, known); cold_extra_captured_at = now()
    completed = now()
    return {'captured_at': captured, 'completed_at': completed,
            'source_registry': safe_string(str(registry)), 'hot_root': safe_string(str(hot)),
            'cold_root': safe_string(str(cold)), 'registered_count': len(rows), 'datasets': rows,
            'additional_top_level_entries': hot_extras, 'hot_extra_captured_at': hot_extra_captured_at,
            'additional_cold_top_level_entries': cold_extras, 'cold_extra_captured_at': cold_extra_captured_at,
            'scope': 'Read-only registry/status/receipt snapshot; not a media or licensing verification.',
            'timestamp_note': 'Fields are read during captured_at–completed_at; source timestamps and dynamic status/receipts may refer to different instants.'}


def write_new_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.catalog-', suffix='.tmp', delete=False) as stream:
            temp_path = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.link(temp_path, path)  # exclusive publication; fails if output exists
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hot-root', type=Path, default=HOT)
    parser.add_argument('--cold-root', type=Path, default=COLD)
    parser.add_argument('--output', type=Path, required=True, help='new JSON file; existing files are not overwritten')
    args = parser.parse_args()
    data = snapshot(args.hot_root, args.cold_root)
    write_new_json(args.output, data)
    print(f"saved {len(data['datasets'])} registered datasets to {args.output}")


if __name__ == '__main__':
    main()
