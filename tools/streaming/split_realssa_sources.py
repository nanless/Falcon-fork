"""Derive reproducible RealSSA video-directory-grouped indices from an audited split.

Output tr/cv/tt are experiment roles, not original official splits.  All three
still point to source_split TARs. Source-directory isolation is not proof of
speaker isolation; overlapping training checkpoints also invalidate holdout
claims. If a present audit row has an unknown source/video identity, that sample is
recorded and excluded. A missing or duplicate audit row rejects the whole
derivation because the audit cannot establish exact index coverage.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def id_digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def derive(index, records, seed=20261112, validation_fraction=0.1, test_fraction=0.1):
    """Pure, deterministic derivation returning indices and auditable manifest."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError('seed must be an integer')
    if not all(math.isfinite(x) and 0 < x < 1 for x in (validation_fraction, test_fraction)):
        raise ValueError('validation and test fractions must be finite and between 0 and 1')
    if validation_fraction + test_fraction >= 1:
        raise ValueError('validation and test fractions must leave training groups')
    samples = index.get('samples')
    source_split = index.get('split')
    if (index.get('format') != 'realssa-tar-index-v1' or source_split != 'tr'
            or not isinstance(samples, list) or index.get('sample_count') != len(samples)
            or index.get('incomplete_count') != 0 or index.get('incomplete') != []
            or not isinstance(index.get('shards'), list)):
        raise ValueError('expected complete official tr index')
    sample_ids = [s.get('id') for s in samples if isinstance(s, dict)]
    if len(sample_ids) != len(samples) or any(not isinstance(x, str) or not x for x in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError('source sample IDs must be unique nonempty strings')

    source_records = [r for r in records if r.get('split') == source_split]
    def valid_hash(value):
        return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
    by_id = {}
    for record in source_records:
        key = record.get('id_sha256')
        if not valid_hash(key):
            raise ValueError('source audit contains invalid sample identity hash')
        if key in by_id:
            raise ValueError('duplicate source audit sample identity')
        by_id[key] = record
    expected = {id_digest(sample_id) for sample_id in sample_ids}
    if len(expected) != len(sample_ids) or set(by_id) != expected:
        raise ValueError('source audit IDs do not exactly match source index')

    known_records = [r for r in source_records
                     if valid_hash(r.get('video_sha256')) and valid_hash(r.get('target_sha256'))]
    target_video = {}
    for record in known_records:
        old = target_video.setdefault(record['target_sha256'], record['video_sha256'])
        if old != record['video_sha256']:
            raise ValueError('same target source mapped to multiple video groups')
    video_groups = sorted({r['video_sha256'] for r in known_records},
                          key=lambda x: (hashlib.sha256(f'{seed}:{x}'.encode()).digest(), x))
    total = len(video_groups)
    num_cv = int(total * validation_fraction)
    num_tt = int(total * test_fraction)
    if num_cv < 1 or num_tt < 1 or num_cv + num_tt >= total:
        raise ValueError('not enough video groups for selected fractions')
    assignment = {group: ('cv' if i < num_cv else 'tt' if i < num_cv + num_tt else 'tr')
                  for i, group in enumerate(video_groups)}
    output_samples = {'tr': [], 'cv': [], 'tt': []}
    output_targets = {k: set() for k in output_samples}
    output_videos = {k: set() for k in output_samples}
    excluded = []
    for sample in samples:
        record = by_id[id_digest(sample['id'])]
        missing = [key for key in ('video_sha256', 'target_sha256') if not valid_hash(record.get(key))]
        if missing:
            excluded.append({'id': sample['id'], 'reason': 'missing_or_invalid_source_identity', 'fields': missing})
            continue
        role = assignment[record['video_sha256']]
        item = dict(sample)
        item['source_group_sha256'] = record['video_sha256']
        item['target_source_sha256'] = record['target_sha256']
        output_samples[role].append(item)
        output_targets[role].add(record['target_sha256'])
        output_videos[role].add(record['video_sha256'])
    for key in ('tr', 'cv', 'tt'):
        if not output_samples[key]:
            raise ValueError('derived empty partition: ' + key)
    for left, right in (('tr','cv'),('tr','tt'),('cv','tt')):
        if output_targets[left] & output_targets[right] or output_videos[left] & output_videos[right]:
            raise AssertionError('source leakage across derived partitions')
    derived = {}
    for role in output_samples:
        result = dict(index)
        result.update(split=role, sample_count=len(output_samples[role]), samples=output_samples[role],
                      source_split=source_split, partition_kind='derived_video_directory_group_holdout')
        derived[role] = result
    manifest = {
        'format': 'falcon-realssa-source-holdout-v1',
        'source_split': source_split,
        'seed': seed,
        'validation_group_fraction': validation_fraction,
        'test_group_fraction': test_fraction,
        'group_assignment': 'sort SHA256(utf8(seed:video_sha256)), assign first cv, next tt, rest tr',
        'caveat': 'video directory and target source path hashes are not verified speaker identities',
        'missing_identity_policy': 'present audit rows with invalid source/video hashes are excluded; missing or duplicate audit rows reject the derivation',
        'warm_start': 'checkpoints trained on full source split have already seen these holdout groups',
        'source_samples': len(samples),
        'excluded_samples': len(excluded),
        'excluded': excluded,
        'source_video_groups': total,
        'partitions': {k: {'samples': len(output_samples[k]), 'video_groups': len(output_videos[k]),
                           'target_sources': len(output_targets[k])} for k in ('tr','cv','tt')},
        'pairwise_source_overlap': {'tr_cv': 0, 'tr_tt': 0, 'cv_tt': 0},
        'video_group_assignments': assignment,
    }
    return derived, manifest


def create(index_path, records_path, audit_path, output_dir, *, seed=20261112,
           validation_fraction=0.1, test_fraction=0.1):
    index_path, records_path, audit_path, output_dir = map(Path, (index_path, records_path, audit_path, output_dir))
    if output_dir.exists():
        raise FileExistsError(output_dir)
    hashes = {'index_sha256': sha256(index_path), 'records_sha256': sha256(records_path),
              'audit_sha256': sha256(audit_path)}
    index = json.loads(index_path.read_text())
    audit = json.loads(audit_path.read_text())
    if audit.get('index_sha256', {}).get('tr') != hashes['index_sha256']:
        raise ValueError('source audit references a different tr index')
    with records_path.open(encoding='utf-8') as source:
        records = [json.loads(line) for line in source if line.strip()]
    for shard in index.get('shards', []):
        st = Path(shard['path']).stat()
        if st.st_size != shard['size'] or st.st_mtime_ns != shard['mtime_ns']:
            raise ValueError('source tar identity changed: ' + shard['path'])
    partitions, manifest = derive(index, records, seed, validation_fraction, test_fraction)
    # Refuse to publish output computed from inputs modified during derivation.
    if hashes != {'index_sha256': sha256(index_path), 'records_sha256': sha256(records_path),
                  'audit_sha256': sha256(audit_path)}:
        raise RuntimeError('source index or audit changed during derivation')
    output_dir.mkdir(parents=False, exist_ok=False)
    manifest['source'] = {'index': str(index_path.resolve()), 'records': str(records_path.resolve()),
                          'audit': str(audit_path.resolve()), **hashes}
    manifest['derived_index_sha256'] = {}
    for role, data in partitions.items():
        path = output_dir / (role + '.json')
        with path.open('x', encoding='utf-8') as target:
            json.dump(data, target, ensure_ascii=False, separators=(',', ':'))
            target.write('\n')
        manifest['derived_index_sha256'][role] = sha256(path)
    with (output_dir / 'manifest.json').open('x', encoding='utf-8') as target:
        json.dump(manifest, target, ensure_ascii=False, indent=2)
        target.write('\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-index', required=True)
    parser.add_argument('--audit-records', required=True)
    parser.add_argument('--audit-summary', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=20261112)
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--test-fraction', type=float, default=0.1)
    args = parser.parse_args()
    result = create(args.train_index, args.audit_records, args.audit_summary, args.output_dir,
                    seed=args.seed, validation_fraction=args.validation_fraction, test_fraction=args.test_fraction)
    print(json.dumps({'output_dir': str(Path(args.output_dir).resolve()),
                      'partitions': result['partitions'],
                      'derived_index_sha256': result['derived_index_sha256']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
