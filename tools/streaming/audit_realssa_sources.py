"""Audit target-source path overlap across RealSSA split indices without copying media.

Output only SHA-256 hashes of sample IDs, source paths and video-directory IDs.
Video directory IDs do not establish speaker identity. Source path equality is
evidence of a common recorded utterance; different paths do not prove absence
of duplicate waveforms. Per-shard caches permit interruption and resumption.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import time


FORMAT = 'realssa-source-audit-v1'


def digest(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def _publish_json(path, obj):
    """Publish complete result atomically, refusing to replace existing files."""
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf8', dir=path.parent,
                                     prefix='.source-audit-', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(obj, stream, ensure_ascii=False, separators=(',', ':'))
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def audit(indices, output_dir, *, workers=4, cache_dir=None):
    """Audit indexed tr/cv/tt metadata; returns summary and saves anonymous evidence."""
    if set(indices) != {'tr', 'cv', 'tt'}:
        raise ValueError('indices must contain exactly tr, cv, tt')
    if not 1 <= workers <= 16:
        raise ValueError('workers must be between 1 and 16')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir) if cache_dir else output / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    report_path = output / 'report.json'
    if report_path.exists():
        raise FileExistsError('audit result already exists')
    datasets, hashes, jobs = {}, {}, []
    for split, path in indices.items():
        data_bytes = Path(path).read_bytes()
        hashes[split] = hashlib.sha256(data_bytes).hexdigest()
        index = json.loads(data_bytes)
        if index.get('format') != 'realssa-tar-index-v1' or index.get('split') != split:
            raise ValueError('invalid index format or split: ' + split)
        if index.get('sample_count') != len(index.get('samples', [])):
            raise ValueError('index sample count mismatch: ' + split)
        datasets[split] = index
        groups = defaultdict(list)
        for row in index['samples']:
            groups[row['shard']].append(row)
        jobs.extend((split, index['shards'][i], rows) for i, rows in groups.items())

    def run(job):
        split, shard, samples = job
        identity = {'format': FORMAT, 'split': split, 'index_sha256': hashes[split],
                    'shard_sha256': digest(shard['path']), 'size': shard['size'],
                    'mtime_ns': shard['mtime_ns'], 'num_records': len(samples)}
        cache_path = cache / (digest(json.dumps(identity, sort_keys=True)) + '.json')
        st = Path(shard['path']).stat()
        if st.st_size != shard['size'] or st.st_mtime_ns != shard['mtime_ns']:
            raise ValueError('indexed shard changed: ' + identity['shard_sha256'])
        if cache_path.exists():
            entry = json.loads(cache_path.read_text())
            if entry.get('identity') != identity:
                raise ValueError('invalid source audit cache')
            return split, entry['rows'], Counter(entry['errors']), len(samples)
        rows, errors = [], Counter()
        with open(shard['path'], 'rb') as stream:
            for sample in samples:
                try:
                    off, size = sample['fields']['meta']
                    stream.seek(off)
                    payload = stream.read(size)
                    if len(payload) != size:
                        raise ValueError('truncated metadata')
                    target = json.loads(payload).get('target')
                    if not isinstance(target, str) or not target:
                        errors['missing_target'] += 1
                        continue
                    target = os.path.normpath(target)
                    video = PurePosixPath(target).parent.name
                    if not re.fullmatch(r'[A-Za-z0-9_-]{11}', video):
                        errors['unrecognized_video_directory'] += 1
                        video = None
                    rows.append({'split': split, 'id_sha256': digest(sample['id']),
                                 'target_sha256': digest(target),
                                 'video_sha256': digest(video) if video else None})
                except Exception as exc:
                    errors['metadata_' + type(exc).__name__] += 1
        after = Path(shard['path']).stat()
        if after.st_size != st.st_size or after.st_mtime_ns != st.st_mtime_ns:
            raise ValueError('indexed shard changed during audit: ' + identity['shard_sha256'])
        entry = {'identity': identity, 'rows': rows, 'errors': dict(errors)}
        try:
            _publish_json(cache_path, entry)
        except FileExistsError:
            pass
        return split, rows, errors, len(samples)

    start = time.monotonic()
    rows, problems, checked = [], defaultdict(Counter), Counter()
    failed_shards = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            split, shard, samples = futures[future]
            try:
                split, found, errors, n = future.result()
            except Exception as exc:
                n = len(samples)
                found = []
                errors = Counter({'shard_' + type(exc).__name__: n})
                failed_shards.append({'split': split, 'shard_sha256': digest(shard['path']),
                                      'error': type(exc).__name__, 'affected_samples': n})
            rows.extend(found)
            problems[split].update(errors)
            checked[split] += n
            if done % 10 == 0 or done == len(jobs):
                print(f'metadata shards {done}/{len(jobs)} elapsed {time.monotonic()-start:.1f}s', flush=True)
    rows.sort(key=lambda r: (r['split'], r['id_sha256'], r['target_sha256']))
    paths, videos, path_counts = defaultdict(set), defaultdict(set), defaultdict(Counter)
    for row in rows:
        paths[row['split']].add(row['target_sha256'])
        path_counts[row['split']][row['target_sha256']] += 1
        if row['video_sha256']:
            videos[row['split']].add(row['video_sha256'])
    pairs = (('tr', 'cv'), ('tr', 'tt'), ('cv', 'tt'))
    summary = {
        'format': FORMAT,
        'scope': 'normalized target-path equality; video directories are not verified speaker IDs',
        'index_sha256': hashes,
        'splits': {split: {'expected': len(datasets[split]['samples']), 'checked': checked[split],
                           'valid_target': sum(r['split'] == split for r in rows),
                           'unique_targets': len(paths[split]), 'unique_video_directories': len(videos[split]),
                           'errors': dict(problems[split])} for split in indices},
        'pairwise': {a+'_'+b: {'target_path_intersection': len(paths[a] & paths[b]),
                                a+'_rows_with_shared_target': sum(path_counts[a][key] for key in paths[a] & paths[b]),
                                b+'_rows_with_shared_target': sum(path_counts[b][key] for key in paths[a] & paths[b]),
                                'video_directory_intersection': len(videos[a] & videos[b])} for a,b in pairs},
        'failed_shards': failed_shards,
        'elapsed_seconds': round(time.monotonic()-start, 3),
    }
    _publish_json(report_path, {'summary': summary, 'records': rows})
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for split in ('tr', 'cv', 'tt'):
        p.add_argument('--' + split + '-index', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--cache-dir')
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    summary = audit({s: getattr(a, s+'_index') for s in ('tr', 'cv', 'tt')},
                    a.output_dir, workers=a.workers, cache_dir=a.cache_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if any(x['errors'] for x in summary['splits'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
