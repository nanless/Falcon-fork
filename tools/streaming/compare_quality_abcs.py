"""Compare matched A/B/C 512-sample evaluations with target-video-group bootstrap."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

from tools.streaming.common import read_checkpoint
from tools.streaming.prepare_quality_abcs import VARIANTS, sha256, verify

PROTOCOL_KEYS = ('scope', 'split', 'source_split', 'partition_kind', 'selection', 'source_provenance_basis',
                 'index_sha256', 'baseline_sha256', 'baseline_epoch',
                 'preprocessing', 'metric', 'short_input_padding', 'code_hash_scope',
                 'code_sha256', 'torch_version', 'device', 'threads')
PAIRS = (('b_causal_variance','a_causal_positive'),
         ('c_centered_variance','b_causal_variance'),
         ('c_centered_variance','a_causal_positive'))


def valid_hash(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def _equal(a,b):
    return math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-7) if finite(a) and finite(b) else a == b


def _checkpoint(variant, entry, manifest, evaluation):
    config = yaml.safe_load(Path(entry['config']).read_text())
    path = Path(config['output']).resolve()
    if Path(evaluation.get('streaming_checkpoint', '')).resolve() != path:
        raise ValueError(variant + ': evaluation does not use configured final.pt')
    before=sha256(path)
    if evaluation.get('streaming_sha256') != before:
        raise ValueError(variant + ': checkpoint digest mismatch')
    ckpt=read_checkpoint(path,'cpu')
    expected_steps=manifest['fixed']['max_steps']
    if ckpt.get('steps') != expected_steps or evaluation.get('streaming_steps') != expected_steps:
        raise ValueError(variant + ': checkpoint is not the fixed-step final output')
    if ckpt.get('configuration_sha256') != entry['sha256']:
        raise ValueError(variant + ': checkpoint training config mismatch')
    effective=ckpt.get('effective_arguments')
    if not isinstance(effective,dict):
        raise ValueError(variant + ': checkpoint lacks effective arguments')
    path_keys={'tar_index','valid_tar_index','visual_init','warm_start','output','checkpoint_dir'}
    for key,value in config.items():
        actual=effective.get(key)
        if key in path_keys and value is not None:
            if not isinstance(actual,str) or Path(actual).resolve()!=Path(value).resolve():
                raise ValueError(variant + ': effective option differs from config: ' + key)
        elif not _equal(actual,value):
            raise ValueError(variant + ': effective option differs from config: ' + key)
    warm=ckpt.get('warm_start_metadata')
    if (not isinstance(warm,dict)
            or warm.get('sha256') != manifest['inputs']['warm_start']['sha256']
            or warm.get('optimizer_scheduler_and_steps_reset') is not True):
        raise ValueError(variant + ': warm-start identity or reset mismatch')
    if (ckpt.get('train_tar_index_sha256') != manifest['inputs']['train_index']['sha256']
            or ckpt.get('validation_tar_index_sha256') != manifest['inputs']['valid_index']['sha256']
            or ckpt.get('visual_initialization_sha256') != manifest['inputs']['visual_init']['sha256']):
        raise ValueError(variant + ': training data or visual initialization mismatch')
    if (ckpt.get('training_status') != 'real_data_optimization' or ckpt.get('world_size') != 2
            or ckpt.get('batch_per_rank') != 1024 or ckpt.get('gradient_accumulation') != 1
            or not _equal(ckpt.get('target_learning_rate'),manifest['fixed']['lr'])):
        raise ValueError(variant + ': training budget or optimizer settings mismatch')
    for key in ('variance_weight','variance_floor','silence_rms_threshold','silence_loss_weight','exclude_degenerate_contrastive'):
        if not _equal(ckpt.get(key), config.get(key)):
            raise ValueError(variant + ': checkpoint option differs from config: ' + key)
    model_config=ckpt.get('model_config',{})
    if model_config.get('visual_mode','causal') != entry['visual_mode'] or evaluation.get('streaming_model_config') != model_config:
        raise ValueError(variant + ': visual mode or model metadata mismatch')
    if ckpt.get('model_code_sha256') != manifest['code_sha256'][str((Path(__file__).resolve().parents[2]/'look2hear/models/FalconStreaming.py').resolve())]:
        raise ValueError(variant + ': model code differs from prepared version')
    protocol=ckpt.get('resume_protocol',{})
    repo=Path(__file__).resolve().parents[2]
    for key,rel in (('trainer_code_sha256','tools/streaming/train_ddp.py'),
                    ('config_parser_sha256','tools/streaming/training_config.py'),
                    ('data_code_sha256','tools/streaming/realssa_tar.py')):
        if protocol.get(key) != manifest['code_sha256'][str((repo/rel).resolve())]:
            raise ValueError(variant + ': training code differs: ' + key)
    if sha256(path)!=before:
        raise RuntimeError(variant + ': checkpoint changed during audit')
    return {'path':str(path),'sha256':before,'steps':ckpt['steps'],
            'config_sha256':ckpt['configuration_sha256'],'model_config':model_config,
            'warm_start_sha256':warm['sha256']}


def compare(experiment_dir, eval_paths, *, seed=20261112, bootstrap=2000):
    """Return exact sample audit, paired means and video-directory cluster CIs."""
    if type(seed) is not int or seed < 0 or type(bootstrap) is not int or bootstrap < 100:
        raise ValueError('seed must be nonnegative and bootstrap must be at least 100')
    experiment_dir=Path(experiment_dir).resolve()
    comparison_path=Path(__file__).resolve()
    comparison_sha=sha256(comparison_path)
    manifest_sha=sha256(experiment_dir/'manifest.json')
    manifest=verify(experiment_dir)
    if set(eval_paths)!=set(VARIANTS):
        raise ValueError('exactly the three named variant evaluations are required')
    eval_paths={k:Path(v).resolve() for k,v in eval_paths.items()}
    eval_hashes={k:sha256(v) for k,v in eval_paths.items()}
    evals={k:json.loads(v.read_text()) for k,v in eval_paths.items()}
    dev_path=Path(manifest['inputs']['valid_index']['path'])
    index=json.loads(dev_path.read_text())
    samples=index['samples'];count=len(samples)
    if count!=512:
        raise ValueError('development set must contain 512 samples')
    if len({x.get('id') for x in samples})!=count:
        raise ValueError('duplicate or invalid development IDs')
    protocol=None; models={}; rows_by_variant={}; baseline_path=None
    repo=Path(__file__).resolve().parents[2]
    eval_code_expected={}
    for absolute,digest in manifest['code_sha256'].items():
        p=Path(absolute)
        try: rel=str(p.relative_to(repo))
        except ValueError: continue
        if (rel.startswith(('look2hear/models/','look2hear/videomodels/','look2hear/layers/',
                            'look2hear/losses/','look2hear/metrics/','look2hear/utils/'))
            or rel in ('tools/streaming/evaluate_realssa.py','tools/streaming/realssa_tar.py',
                       'tools/streaming/common.py','configs/Falcon.yml')):
            eval_code_expected[rel]=digest
    for name,entry in manifest['variants'].items():
        e=evals[name]
        if not isinstance(e,dict) or any(key not in e for key in PROTOCOL_KEYS):
            raise ValueError(name + ': missing evaluation protocol')
        current={key:e[key] for key in PROTOCOL_KEYS}
        if protocol is None: protocol=current
        elif current!=protocol:
            raise ValueError(name + ': evaluation protocol differs')
        if (e['index_sha256']!=manifest['inputs']['valid_index']['sha256']
                or e['split']!='cv' or e['code_sha256']!=eval_code_expected):
            raise ValueError(name + ': index or evaluation code digest mismatch')
        if (e['source_split'] != index.get('source_split')
                or e['partition_kind'] != index.get('partition_kind')
                or e['selection'] != index.get('selection')
                or e['source_provenance_basis'] != 'explicit'):
            raise ValueError(name + ': derived development provenance mismatch')
        if e.get('start')!=0 or e.get('end_exclusive')!=count:
            raise ValueError(name + ': evaluation does not cover entire dev index')
        candidate_baseline=Path(e.get('baseline_checkpoint','')).resolve()
        if baseline_path is None:baseline_path=candidate_baseline
        elif candidate_baseline!=baseline_path:raise ValueError('baseline paths differ')
        if sha256(candidate_baseline)!=e['baseline_sha256']:
            raise ValueError(name + ': baseline checkpoint digest mismatch')
        models[name]=_checkpoint(name,entry,manifest,e)
        rows=e.get('rows')
        if not isinstance(rows,list) or len(rows)!=count:
            raise ValueError(name + ': incorrect row count')
        positions={}
        for row in rows:
            i=row.get('index') if isinstance(row,dict) else None
            if type(i) is not int or not 0<=i<count or i in positions:
                raise ValueError(name + ': duplicate/out-of-range sample index')
            if row.get('id')!=samples[i]['id']:
                raise ValueError(name + ': sample ID mismatch')
            if row.get('status') not in ('ok','error','invalid_metric'):
                raise ValueError(name + ': unrecognized sample status')
            if row['status']=='ok' and (not finite(row.get('falcon_si_sdri_db')) or not finite(row.get('streaming_si_sdri_db'))):
                raise ValueError(name + ': invalid finite metric')
            positions[i]=row
        valid=sum(row['status']=='ok' for row in positions.values())
        summary=e.get('summary',{})
        if summary.get('attempted')!=count or summary.get('valid_pairs')!=valid or summary.get('failed')!=count-valid:
            raise ValueError(name + ': evaluation summary counts do not match rows')
        rows_by_variant[name]=positions
    names=list(VARIANTS)
    reference=rows_by_variant[names[0]]
    valid_set={i for i,row in reference.items() if row['status']=='ok'}
    for name in names[1:]:
        if {i for i,row in rows_by_variant[name].items() if row['status']=='ok'}!=valid_set:
            raise ValueError(name + ': valid-sample set differs')
    invalid=[];excluded=[];included=[];baseline_max_spread=0.0
    for i in range(count):
        per={name:rows_by_variant[name][i] for name in names}
        if i not in valid_set:
            invalid.append({'index':i,'id':samples[i]['id'],
                            'results':{name:{'status':r['status'],'error':r.get('error')} for name,r in per.items()}})
            continue
        baseline=[per[name]['falcon_si_sdri_db'] for name in names]
        baseline_max_spread=max(baseline_max_spread,max(baseline)-min(baseline))
        if max(baseline)-min(baseline)>0.001:
            raise ValueError('per-sample baseline scores differ: ' + str(i))
        group=samples[i].get('source_group_sha256')
        target_source=samples[i].get('target_source_sha256')
        if not valid_hash(group) or not valid_hash(target_source):
            excluded.append({'index':i,'id':samples[i]['id'],'reason':'missing_or_invalid_video_or_target_source_hash'})
            continue
        scores={name:per[name]['streaming_si_sdri_db'] for name in names}
        included.append({'index':i,'id':samples[i]['id'],'video_group_sha256':group,
                         'falcon_si_sdri_db':baseline[0], 'scores':scores})
    if not included:
        raise ValueError('no valid samples with source group')
    groups=sorted({r['video_group_sha256'] for r in included})
    if len(groups)<2:
        raise ValueError('at least two source groups are needed for cluster bootstrap')
    group_index={g:i for i,g in enumerate(groups)}
    group_counts=np.zeros(len(groups),dtype=np.int64)
    group_sums={f'{a}_minus_{b}':np.zeros(len(groups),dtype=np.float64) for a,b in PAIRS}
    for row in included:
        idx=group_index[row['video_group_sha256']];group_counts[idx]+=1
        for a,b in PAIRS:group_sums[f'{a}_minus_{b}'][idx]+=row['scores'][a]-row['scores'][b]
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(groups),size=(bootstrap,len(groups)))
    sampled_counts=group_counts[draws].sum(axis=1)
    pairs={}
    for a,b in PAIRS:
        key=f'{a}_minus_{b}'; gs=group_sums[key]
        distribution=gs[draws].sum(axis=1)/sampled_counts
        delta=[r['scores'][a]-r['scores'][b] for r in included]
        low,high=np.quantile(distribution,[.025,.975])
        pairs[key]={'mean_db':float(np.mean(delta)), 'median_db':float(np.median(delta)),
                    'bootstrap_95_ci_db':[float(low),float(high)],
                    'numerator_better_n':sum(x>0 for x in delta)}
    means={name:float(np.mean([r['scores'][name] for r in included])) for name in names}
    means['falcon']=float(np.mean([r['falcon_si_sdri_db'] for r in included]))
    # Verify that raw evals, baseline and all prepared sources are stable throughout.
    if any(sha256(path)!=eval_hashes[name] for name,path in eval_paths.items()):
        raise RuntimeError('evaluation input changed during comparison')
    if sha256(baseline_path)!=protocol['baseline_sha256']:
        raise RuntimeError('baseline checkpoint changed during comparison')
    verify(experiment_dir)
    if sha256(experiment_dir/'manifest.json')!=manifest_sha:
        raise RuntimeError('experiment manifest changed during comparison')
    if sha256(comparison_path)!=comparison_sha:
        raise RuntimeError('comparison code changed during comparison')
    return {'format':'falcon-quality-abcs-paired-v1',
            'scope':'paired central-2s ch0 on derived tr development; checkpoint pretrained on full tr',
            'source_note':'target video-directory groups only; not verified speakers; warm-start saw all original tr',
            'bootstrap':{'unit':'target video directory hash','seed':seed,'replicates':bootstrap,
                         'interval':'percentile 2.5/97.5','weighting':'resample groups then average all selected samples'},
            'counts':{'indexed':count,'valid_before_group_check':len(valid_set),'invalid':len(invalid),
                      'excluded_missing_group':len(excluded),'paired':len(included),'video_groups':len(groups)},
            'comparison_code':str(comparison_path), 'comparison_code_sha256':comparison_sha,
            'numpy_version':str(np.__version__),
            'experiment_manifest_sha256':manifest_sha,
            'input_index_sha256':manifest['inputs']['valid_index']['sha256'],
            'baseline_checkpoint':str(baseline_path),'baseline_sha256':protocol['baseline_sha256'],
            'evaluations':{name:{'path':str(path),'sha256':eval_hashes[name]} for name,path in eval_paths.items()},
            'baseline_max_score_spread_db':baseline_max_spread,
            'models':models,'means_si_sdri_db':means,'pairs':pairs,
            'invalid':invalid,'excluded':excluded,'rows':included}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-dir',required=True)
    p.add_argument('--a',required=True);p.add_argument('--b',required=True);p.add_argument('--c',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=20261112)
    p.add_argument('--bootstrap',type=int,default=2000)
    a=p.parse_args(); out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    values={'a_causal_positive':a.a,'b_causal_variance':a.b,'c_centered_variance':a.c}
    result=compare(a.experiment_dir,values,seed=a.seed,bootstrap=a.bootstrap)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'counts':result['counts'],'means':result['means_si_sdri_db'],'pairs':result['pairs']},ensure_ascii=False))


if __name__=='__main__': main()
