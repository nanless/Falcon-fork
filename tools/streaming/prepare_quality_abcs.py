"""Prepare, never launch, three matched warm-start RealSSA diagnostics.

A: causal + positive only; B: causal + variance; C: block-centered + variance.
The common checkpoint was already trained on the full original tr corpus, so
the derived group holdout has been seen and these are not unseen-source results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import yaml

from tools.streaming.training_config import UniqueKeyLoader
from tools.streaming.common import read_checkpoint

VARIANTS = {
    'a_causal_positive': ('causal', 0.0),
    'b_causal_variance': ('causal', None),
    'c_centered_variance': ('chunk_centered', None),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _load_derived(path, role):
    obj = json.loads(Path(path).read_text())
    samples = obj.get('samples')
    if (obj.get('format') != 'realssa-tar-index-v1' or obj.get('split') != role
            or obj.get('source_split') != 'tr'
            or obj.get('partition_kind') != 'derived_video_directory_group_holdout'
            or not isinstance(samples, list) or len(samples) != obj.get('sample_count') or not samples):
        raise ValueError('expected explicit tr-derived ' + role + ' index')
    ids = [x.get('id') for x in samples]
    if len(set(ids)) != len(ids) or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError('invalid/duplicate sample IDs')
    for item in samples:
        for key in ('source_group_sha256', 'target_source_sha256'):
            value = item.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('missing source identity: ' + key)
    return obj


def prepare(base_config, train_index, valid_index, warm_start, holdout_manifest, output_dir, *,
            max_steps=50, lr=6.25e-5, warmup_steps=0, seed=42, variance_weight=0.1):
    """Prepare configs and immutable identity manifest; no model execution."""
    if type(max_steps) is not int or max_steps < 1 or type(warmup_steps) is not int or warmup_steps < 0:
        raise ValueError('max_steps must be positive; warmup_steps nonnegative')
    if type(seed) is not int or not 0 <= seed <= 2**32 - 2:
        raise ValueError('invalid two-rank seed')
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(variance_weight) or variance_weight <= 0:
        raise ValueError('learning rate and experimental variance weight must be finite and positive')
    base_config, train_index, valid_index, warm_start, holdout_manifest, output_dir = (
        Path(x).expanduser().resolve() for x in (base_config, train_index, valid_index, warm_start, holdout_manifest, output_dir))
    if output_dir.exists():
        raise FileExistsError(output_dir)
    paths = {'base_config': base_config, 'train_index': train_index, 'valid_index': valid_index,
             'warm_start': warm_start, 'holdout_manifest': holdout_manifest}
    identities = {key: {'path': str(path), 'sha256': sha256(path)} for key,path in paths.items()}
    with base_config.open(encoding='utf-8') as stream:
        base = yaml.load(stream, Loader=UniqueKeyLoader)
    if not isinstance(base, dict):
        raise ValueError('base config must be a mapping')
    train = _load_derived(train_index, 'tr')
    valid = _load_derived(valid_index, 'cv')
    holdout = json.loads(holdout_manifest.read_text())
    if holdout.get('format') != 'falcon-realssa-source-holdout-v1':
        raise ValueError('invalid source holdout manifest')
    if holdout.get('derived_index_sha256', {}).get('tr') != identities['train_index']['sha256']:
        raise ValueError('training index does not match holdout manifest')
    cv_parent = holdout_manifest.parent / 'cv.json'
    identities['valid_parent_index'] = {'path': str(cv_parent), 'sha256': sha256(cv_parent)}
    if identities['valid_parent_index']['sha256'] != holdout.get('derived_index_sha256', {}).get('cv'):
        raise ValueError('parent development index does not match holdout manifest')
    full_valid = _load_derived(cv_parent, 'cv')
    by_id = {item['id']: item for item in full_valid['samples']}
    if any(by_id.get(item['id']) != item for item in valid['samples']):
        raise ValueError('fast development rows differ from parent index')
    selection = valid.get('selection', {})
    if (selection.get('role') != 'fast_dev_from_source_tr'
            or selection.get('parent_index_sha256') != holdout.get('derived_index_sha256', {}).get('cv')
            or valid['sample_count'] != 512):
        raise ValueError('validation index is not the requested 512-item derived development subset')
    train_groups = {x['source_group_sha256'] for x in train['samples']}
    valid_groups = {x['source_group_sha256'] for x in valid['samples']}
    train_sources = {x['target_source_sha256'] for x in train['samples']}
    valid_sources = {x['target_source_sha256'] for x in valid['samples']}
    if train_groups & valid_groups or train_sources & valid_sources:
        raise ValueError('derived train/development sources overlap')

    visual_path = base.get('visual_init')
    if not isinstance(visual_path, str) or not visual_path:
        raise ValueError('base config requires visual_init')
    visual_path = Path(visual_path).expanduser()
    visual_path = (base_config.parent / visual_path).resolve() if not visual_path.is_absolute() else visual_path.resolve()
    identities['visual_init'] = {'path': str(visual_path), 'sha256': sha256(visual_path)}
    source_checkpoint = read_checkpoint(str(warm_start), 'cpu')
    source_config = source_checkpoint['model_config']
    if (source_config['channels'] != base.get('model_channels')
            or tuple(source_config['dilations']) != (1,2,4,8,16,32)
            or source_config.get('visual_mode', 'causal') != 'causal'
            or source_checkpoint.get('speaker_pool_state') is None
            or source_checkpoint.get('visual_initialization_sha256') != identities['visual_init']['sha256']
            or source_checkpoint.get('train_tar_index_sha256') != holdout.get('source', {}).get('index_sha256')):
        raise ValueError('warm-start checkpoint is not the expected full-tr causal model and visual initialization')
    repo = Path(__file__).resolve().parents[2]
    code_paths = [repo/'tools/streaming/train_ddp.py', repo/'tools/streaming/training_config.py',
                  repo/'tools/streaming/realssa_tar.py', repo/'tools/streaming/common.py',
                  repo/'tools/streaming/evaluate_realssa.py', repo/'configs/Falcon.yml', Path(__file__).resolve()]
    for directory in ('models', 'videomodels', 'layers', 'losses', 'metrics', 'utils'):
        code_paths.extend(sorted((repo/'look2hear'/directory).rglob('*.py')))
    code_hashes = {str(p.resolve()): sha256(p) for p in sorted(set(code_paths))}

    common = dict(base)
    for key in ('index_dir','synthetic','valid_index_dir','random_visual','reconstruction_only','resume','warm_start'):
        common.pop(key, None)
    common.update(tar_index=str(train_index), valid_tar_index=str(valid_index),
                  visual_init=str(visual_path), warm_start=str(warm_start),
                  positive_contrastive=True, variance_floor=0.05, silence_rms_threshold=0.0,
                  silence_loss_weight=1.0, exclude_degenerate_contrastive=True,
                  batch_size=1024, accum=1, max_steps=max_steps, lr=float(lr),
                  warmup_steps=warmup_steps, seed=seed, validation_limit=0)
    output_dir.mkdir(parents=False, exist_ok=False)
    variants = {}
    for name, (mode, weight) in VARIANTS.items():
        config = dict(common, visual_mode=mode, variance_weight=variance_weight if weight is None else weight,
                      output=str(output_dir/name/'final.pt'), checkpoint_dir=str(output_dir/name/'epochs'))
        path = output_dir/(name + '.yml')
        with path.open('x', encoding='utf-8') as stream:
            yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
        variants[name] = {'config': str(path), 'sha256': sha256(path), 'visual_mode': mode,
                          'variance_weight': config['variance_weight']}
    if any(sha256(v['path']) != v['sha256'] for v in identities.values()) or any(sha256(p) != h for p,h in code_hashes.items()):
        raise RuntimeError('inputs or code changed while preparing experiments')
    manifest = {
        'format': 'falcon-quality-abcs-v1',
        'description': 'A vs B tests variance regularization; B vs C tests visual context. All use same silence handling.',
        'warning': 'The fixed warm-start checkpoint was trained on the full original tr including derived development and holdout sources; these diagnostics are not unseen-source generalization.',
        'data_roles': {'train': train['sample_count'], 'development': valid['sample_count'],
                       'both_original_source_split': 'tr', 'holdout_tt_not_opened_or_evaluated': True},
        'warm_start_source': {'format': source_checkpoint['format'], 'steps': source_checkpoint.get('steps'),
                              'model_config': source_config,
                              'train_tar_index_sha256': source_checkpoint.get('train_tar_index_sha256')},
        'inputs': identities, 'code_sha256': code_hashes,
        'fixed': {'max_steps': max_steps, 'batch_per_gpu': 1024, 'world_size': 2, 'accum': 1,
                  'lr': lr, 'warmup_steps': warmup_steps, 'seed': seed,
                  'variance_floor': 0.05, 'silence_rms_threshold': 0.0, 'silence_loss_weight': 1.0,
                  'exclude_degenerate_contrastive': True},
        'variants': variants,
        'launch_example': 'PYTHONPATH=. torchrun --standalone --nproc_per_node=2 -m tools.streaming.train_ddp --config <variant.yml>',
    }
    with (output_dir/'manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return manifest


def verify(output_dir):
    output_dir=Path(output_dir).resolve()
    manifest=json.loads((output_dir/'manifest.json').read_text())
    if manifest.get('format') != 'falcon-quality-abcs-v1':
        raise ValueError('invalid experiment manifest')
    for v in manifest['inputs'].values():
        if sha256(v['path']) != v['sha256']:
            raise ValueError('prepared input changed: ' + v['path'])
    for path, digest in manifest['code_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('prepared source changed: ' + path)
    for variant in manifest['variants'].values():
        if sha256(variant['config']) != variant['sha256']:
            raise ValueError('prepared config changed: ' + variant['config'])
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-config')
    p.add_argument('--train-index')
    p.add_argument('--valid-index')
    p.add_argument('--warm-start')
    p.add_argument('--holdout-manifest')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--max-steps', type=int, default=50)
    p.add_argument('--lr', type=float, default=6.25e-5)
    p.add_argument('--warmup-steps', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--variance-weight', type=float, default=0.1)
    p.add_argument('--verify', action='store_true')
    args=p.parse_args()
    if args.verify:
        result=verify(args.output_dir)
        print(json.dumps({'verified': True, 'output_dir': str(Path(args.output_dir).resolve())}))
    else:
        if not all((args.base_config,args.train_index,args.valid_index,args.warm_start,args.holdout_manifest)):
            p.error('preparation requires --base-config, --train-index, --valid-index, --warm-start and --holdout-manifest')
        result=prepare(args.base_config,args.train_index,args.valid_index,args.warm_start,args.holdout_manifest,args.output_dir,
                       max_steps=args.max_steps,lr=args.lr,warmup_steps=args.warmup_steps,seed=args.seed,variance_weight=args.variance_weight)
        print(json.dumps({'output_dir': str(Path(args.output_dir).resolve()), 'variants': result['variants']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
