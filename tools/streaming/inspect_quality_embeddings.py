"""Compare checkpoint training-head retrieval on a fixed source-grouped CV subset.

Only a 512-row development index derived from official RealSSA tr is accepted.
One nondegenerate center-cropped sample is selected from each of 48 different
source-directory hashes; different source directories are not verified speakers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from look2hear.models.FalconStreaming import FalconStreaming
from tools.streaming.common import read_checkpoint
from tools.streaming.realssa_tar import RealSSATarDataset
from tools.streaming.train_ddp import TrainingWrapper, pad_two_seconds, target_degeneracy_masks


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_development_index(index, parent, manifest, parent_sha):
    """Ensure this is an exact 512-row derived CV subset, never a TT index."""
    if (not isinstance(manifest, dict) or manifest.get('format') != 'falcon-realssa-source-holdout-v1'
            or manifest.get('source_split') != 'tr'
            or not isinstance(manifest.get('derived_index_sha256'), dict)
            or manifest['derived_index_sha256'].get('cv') != parent_sha):
        raise ValueError('holdout manifest does not identify the parent cv index')
    if not isinstance(index, dict) or not isinstance(parent, dict):
        raise ValueError('indices must be JSON objects')
    for item in (index, parent):
        if (item.get('format') != 'realssa-tar-index-v1' or item.get('split') != 'cv'
                or item.get('source_split') != 'tr'
                or item.get('partition_kind') != 'derived_video_directory_group_holdout'
                or not isinstance(item.get('samples'), list)
                or item.get('sample_count') != len(item['samples'])
                or item.get('incomplete_count') != 0 or item.get('incomplete') != []):
            raise ValueError('expected a complete source-tr-derived cv index')
    selection = index.get('selection')
    if (index['sample_count'] != 512 or not isinstance(selection, dict)
            or selection.get('role') != 'fast_dev_from_source_tr'
            or selection.get('parent_index_sha256') != parent_sha):
        raise ValueError('expected the 512-row fast development subset of parent cv')
    if not isinstance(parent.get('shards'), list) or index.get('shards') != parent['shards']:
        raise ValueError('development shard identities differ from parent cv')
    parent_rows = parent['samples']
    if selection.get('parent_sample_count') != len(parent_rows):
        raise ValueError('development parent sample count mismatch')
    parent_ids = [row.get('id') for row in parent_rows if isinstance(row, dict)]
    if (len(parent_ids) != len(parent_rows) or any(not isinstance(s, str) or not s for s in parent_ids)
            or len(set(parent_ids)) != len(parent_ids)):
        raise ValueError('parent cv contains invalid or duplicated sample IDs')
    by_id = {row['id']: row for row in parent_rows}
    group_roles = manifest.get('video_group_assignments')
    if not isinstance(group_roles, dict):
        raise ValueError('manifest lacks source group assignments')
    ids = set()
    for row in index['samples']:
        if not isinstance(row, dict):
            raise ValueError('development sample must be an object')
        name, group, target = row.get('id'), row.get('source_group_sha256'), row.get('target_source_sha256')
        if (not isinstance(name, str) or not name or name in ids or by_id.get(name) != row
                or not valid_hash(group) or not valid_hash(target) or group_roles.get(group) != 'cv'):
            raise ValueError('development row identity, source group or parent mismatch')
        ids.add(name)
    return index['samples']


def candidate_groups(samples, seed):
    """Deterministic source-group and within-group order, independent of models."""
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    grouped = {}
    for i, row in enumerate(samples):
        grouped.setdefault(row['source_group_sha256'], []).append(i)
    def order(text):
        return hashlib.sha256(f'{seed}:{text}'.encode('utf-8')).digest()
    return [(group, sorted(grouped[group], key=lambda i: (order(samples[i]['id']), samples[i]['id'])))
            for group in sorted(grouped, key=lambda group: (order(group), group))]


def select_samples(samples, loader, *, seed, count=48, rms_threshold=0.0):
    """Select at most one valid example per group; data/target determines selection."""
    if (type(count) is not int or count < 1 or type(rms_threshold) not in (int, float)
            or not math.isfinite(rms_threshold) or rms_threshold < 0):
        raise ValueError('invalid sample count or target RMS threshold')
    selected, skipped = [], []
    for group, candidates in candidate_groups(samples, seed):
        for i in candidates:
            try:
                mix, target, video = loader(i)
            except (OSError, ValueError, RuntimeError) as exc:
                skipped.append({'index': i, 'id': samples[i]['id'], 'reason': type(exc).__name__})
                continue
            if (not all(isinstance(x, torch.Tensor) and torch.isfinite(x).all() for x in (mix, target, video))
                    or tuple(mix.shape) != (32000,) or tuple(target.shape) != (32000,)
                    or tuple(video.shape) != (50, 88, 88)):
                skipped.append({'index': i, 'id': samples[i]['id'], 'reason': 'invalid_tensor'})
                continue
            degenerate, _ = target_degeneracy_masks(target[None], rms_threshold)
            if bool(degenerate[0]):
                skipped.append({'index': i, 'id': samples[i]['id'], 'reason': 'degenerate_target'})
                continue
            selected.append({'index': i, 'id': samples[i]['id'],
                             'source_group_sha256': group,
                             'target_source_sha256': samples[i]['target_source_sha256'],
                             'tensors': (mix, target, video)})
            break
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f'only {len(selected)} nondegenerate distinct source groups found; need {count}')
    return selected, skipped


def retrieval_summary(q, p):
    """Exact sample-pair retrieval; stable ties use selected-sample order."""
    if (not isinstance(q, torch.Tensor) or not isinstance(p, torch.Tensor)
            or q.ndim != 2 or q.shape != p.shape or q.shape[0] < 2
            or not torch.isfinite(q).all() or not torch.isfinite(p).all()):
        raise ValueError('q and p must be matching finite [samples,dimensions] tensors')
    q, p = q.detach().to('cpu', torch.float64), p.detach().to('cpu', torch.float64)
    qnorm, pnorm = q.norm(dim=1), p.norm(dim=1)
    if (qnorm <= 1e-12).any() or (pnorm <= 1e-12).any():
        raise ValueError('zero-norm embeddings cannot be ranked')
    q, p = F.normalize(q, dim=1), F.normalize(p, dim=1)
    sim = q @ p.T
    n = q.shape[0]
    order = torch.argsort(sim, dim=1, descending=True, stable=True)
    correct = torch.arange(n).unsqueeze(1)
    ranks = (order == correct).nonzero(as_tuple=False)[:, 1] + 1
    target_sim = p @ p.T
    offdiag = ~torch.eye(n, dtype=torch.bool)
    diagonal = sim.diag()
    wrong = sim[offdiag]
    tied = ((sim == diagonal[:, None]).sum(dim=1) > 1).sum().item()
    return {
        'n': n, 'top1_count': int((ranks == 1).sum()),
        'top5_count': int((ranks <= min(5, n)).sum()),
        'top1_fraction': float((ranks == 1).to(torch.float64).mean()),
        'top5_fraction': float((ranks <= min(5, n)).to(torch.float64).mean()),
        'mean_rank': float(ranks.to(torch.float64).mean()), 'ranks': ranks.tolist(),
        'correct_pair_cosine_mean': float(diagonal.mean()),
        'wrong_pair_cosine_mean': float(wrong.mean()),
        'target_cross_cosine_mean': float(target_sim[offdiag].mean()),
        'target_cross_cosine_median': float(torch.quantile(target_sim[offdiag], .5)),
        'target_unit_vector_per_dimension_std_mean': float(p.std(dim=0, unbiased=False).mean()),
        'ties_to_correct_count': int(tied),
        'tie_policy': 'stable descending score, ties resolved by selected-sample order',
        'random_order_expectation': {'top1_fraction': 1 / n, 'top5_fraction': min(5, n) / n,
                                     'mean_rank': (n + 1) / 2},
    }


def inspect_checkpoint(path, selected, device, batch_size):
    path = Path(path).resolve()
    before = sha256(path)
    ckpt = read_checkpoint(path, 'cpu')
    if not isinstance(ckpt.get('speaker_pool_state'), dict):
        raise ValueError(f'checkpoint has no trained speaker_pool_state: {path}')
    model = FalconStreaming(**ckpt['model_config'])
    model.load_state_dict(ckpt['state_dict'], strict=True)
    wrapper = TrainingWrapper(model, True)
    wrapper.speaker_pool.load_state_dict(ckpt['speaker_pool_state'], strict=True)
    wrapper = wrapper.to(device).eval()
    queries, positives = [], []
    with torch.inference_mode():
        for start in range(0, len(selected), batch_size):
            items = selected[start:start + batch_size]
            mixture = torch.stack([r['tensors'][0] for r in items]).to(device)
            target = torch.stack([r['tensors'][1] for r in items]).to(device)
            video = torch.stack([r['tensors'][2] for r in items]).to(device)
            padded, frames = pad_two_seconds(mixture, video)
            output = wrapper.model(padded, frames)[:, :32000]
            q = wrapper.speaker_pool(wrapper.encode_waveform(output))
            p = wrapper.speaker_pool(wrapper.encode_waveform(target))
            queries.append(q.cpu())
            positives.append(p.cpu())
    if sha256(path) != before:
        raise RuntimeError(f'checkpoint changed during inspection: {path}')
    result = retrieval_summary(torch.cat(queries), torch.cat(positives))
    result.update({'checkpoint': str(path), 'sha256': before,
                   'steps': ckpt.get('steps'), 'training_status': ckpt.get('training_status'),
                   'model_config': ckpt['model_config'],
                   'train_index_sha256': ckpt.get('train_tar_index_sha256'),
                   'validation_index_sha256': ckpt.get('validation_tar_index_sha256'),
                   'warm_start': ckpt.get('warm_start_metadata')})
    return result


def run(index_path, manifest_path, checkpoints, *, seed=20261112, device='cpu', batch_size=4,
        rms_threshold=0.0):
    index_path, manifest_path = Path(index_path).resolve(), Path(manifest_path).resolve()
    parent_path = manifest_path.parent / 'cv.json'
    inputs = (index_path, parent_path, manifest_path)
    hashes = {str(p): sha256(p) for p in inputs}
    index = json.loads(index_path.read_text())
    parent = json.loads(parent_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    samples = validate_development_index(index, parent, manifest, hashes[str(parent_path)])
    if not checkpoints:
        raise ValueError('at least one checkpoint is required')
    if device not in ('cpu', 'cuda') or type(batch_size) is not int or batch_size < 1:
        raise ValueError('device must be cpu/cuda and batch-size must be positive')
    repo = Path(__file__).resolve().parents[2]
    code_paths = [Path(__file__).resolve(), repo/'tools/streaming/train_ddp.py',
                  repo/'tools/streaming/common.py', repo/'tools/streaming/realssa_tar.py']
    for folder in ('models', 'videomodels', 'layers', 'losses', 'utils'):
        code_paths.extend(sorted((repo/'look2hear'/folder).rglob('*.py')))
    code_hashes = {str(p.relative_to(repo)): sha256(p) for p in code_paths}
    data = RealSSATarDataset(str(index_path), seed=42, crop='center', augment=False)
    selected, skipped = select_samples(samples, data.__getitem__, seed=seed,
                                       count=48, rms_threshold=rms_threshold)
    selected_shards = sorted({samples[r['index']]['shard'] for r in selected})
    model_results = [inspect_checkpoint(path, selected, device, batch_size) for path in checkpoints]
    for i in selected_shards:
        entry = data.shards[i]
        stat = Path(entry['path']).stat()
        if stat.st_size != entry['size'] or stat.st_mtime_ns != entry['mtime_ns']:
            raise RuntimeError('selected source shard changed during inspection')
    if any(sha256(repo/path) != digest for path, digest in code_hashes.items()):
        raise RuntimeError('inspection code changed during run')
    if any(sha256(p) != digest for p, digest in hashes.items()):
        raise RuntimeError('index or holdout manifest changed during inspection')
    return {
        'format': 'falcon-sourcegroup-retrieval-v1',
        'scope': ('48 distinct source-directory groups in source-tr-derived CV; '
                  'source groups are not verified speaker identities; warm-start source may have seen them'),
        'selection': 'SHA256(seed:source_group) order then SHA256(seed:sample_id) order; '
                     'first decodable nondegenerate target per group',
        'seed': seed, 'target_rms_threshold': rms_threshold,
        'index': str(index_path), 'index_sha256': hashes[str(index_path)],
        'parent_index': str(parent_path), 'parent_index_sha256': hashes[str(parent_path)],
        'holdout_manifest': str(manifest_path), 'holdout_manifest_sha256': hashes[str(manifest_path)],
        'selected': [{key: row[key] for key in ('index','id','source_group_sha256','target_source_sha256')}
                     for row in selected], 'skipped': skipped,
        'device': device, 'torch_version': str(torch.__version__), 'batch_size': batch_size,
        'selected_shards': selected_shards, 'source_shard_identity': 'indexed size and mtime, not content hash',
        'code_sha256': code_hashes,
        'target_mode': 'p from target, q from model output; shared audio projection and speaker pool; L2-normalized',
        'results': model_results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True)
    parser.add_argument('--holdout-manifest', required=True)
    parser.add_argument('--checkpoint', required=True, nargs='+')
    parser.add_argument('--seed', type=int, default=20261112)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--target-rms-threshold', type=float, default=0.)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(args.threads)
    result = run(args.index, args.holdout_manifest, args.checkpoint, seed=args.seed,
                 device=args.device, batch_size=args.batch_size, rms_threshold=args.target_rms_threshold)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps([{'checkpoint': r['checkpoint'], 'top1_count': r['top1_count'],
                       'top5_count': r['top5_count'], 'mean_rank': r['mean_rank']}
                      for r in result['results']], ensure_ascii=False))


if __name__ == '__main__':
    main()
