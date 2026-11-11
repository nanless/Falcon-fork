"""Prepare and audit the reconstruction-only D control; never launch work.

D differs from A only by disabling the positive contrastive objective. Both
warm-start on the full original training set, including the derived dev sources.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from tools.streaming.common import read_checkpoint
from tools.streaming.prepare_quality_abcs import sha256, verify as verify_abc
from tools.streaming.compare_quality_abcs import PROTOCOL_KEYS, _checkpoint, finite, valid_hash

A = 'a_causal_positive'


def prepare(abc_dir, output_dir):
    abc_dir, output_dir = Path(abc_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    abc = verify_abc(abc_dir)
    source_manifest_sha = sha256(abc_dir / 'manifest.json')
    entry = abc['variants'][A]
    config = yaml.safe_load(Path(entry['config']).read_text())
    if (config.get('positive_contrastive') is not True or config.get('variance_weight') != 0
        or config.get('visual_mode') != 'causal' or config.get('max_steps') != 50):
        raise ValueError('A is not the requested 50-step causal positive-only control')
    if not output_dir.parent.is_dir():
        raise ValueError('output parent must already exist')
    d = dict(config, positive_contrastive=False, reconstruction_only=True,
             output=str(output_dir / 'final.pt'), checkpoint_dir=str(output_dir / 'epochs'))
    output_dir.mkdir(exist_ok=False)
    config_path = output_dir / 'd_reconstruction.yml'
    with config_path.open('x') as f:
        yaml.safe_dump(d, f, allow_unicode=True, sort_keys=False)
    source_files = [Path(__file__).resolve(), Path(__file__).resolve().with_name('compare_quality_abcs.py')]
    manifest = {'format':'falcon-quality-d-v1', 'abc_dir':str(abc_dir),
                'abc_manifest_sha256':source_manifest_sha, 'a_config':entry,
                'd_config':{'path':str(config_path),'sha256':sha256(config_path)},
                'tool_sha256':{str(p):sha256(p) for p in source_files},
                'factor':'positive_contrastive: true to false, reconstruction_only: true; variance remains 0',
                'warm_start_note':'full original tr, including development sources, was previously seen',
                'data_note':'same original-tr-derived cv_fast512; video directories are not verified speakers'}
    verify_abc(abc_dir)
    if sha256(abc_dir / 'manifest.json') != source_manifest_sha:
        raise RuntimeError('ABC manifest changed during preparation')
    with (output_dir / 'manifest.json').open('x') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2); f.write('\n')
    return manifest


def verify_d(output_dir):
    root = Path(output_dir).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('format') != 'falcon-quality-d-v1':
        raise ValueError('invalid D manifest')
    abc_dir = Path(manifest['abc_dir'])
    if sha256(abc_dir / 'manifest.json') != manifest['abc_manifest_sha256']:
        raise ValueError('ABC manifest changed')
    abc = verify_abc(abc_dir)
    if abc['variants'][A] != manifest['a_config']:
        raise ValueError('A config identity changed')
    for p, digest in manifest['tool_sha256'].items():
        if sha256(p) != digest: raise ValueError('D audit tool changed: ' + p)
    dpath = root / 'd_reconstruction.yml'
    if str(dpath) != manifest['d_config']['path'] or sha256(dpath) != manifest['d_config']['sha256']:
        raise ValueError('D config changed')
    a = yaml.safe_load(Path(manifest['a_config']['config']).read_text())
    d = yaml.safe_load(dpath.read_text())
    expected = dict(a, positive_contrastive=False, reconstruction_only=True, output=str(root/'final.pt'), checkpoint_dir=str(root/'epochs'))
    if d != expected or a.get('positive_contrastive') is not True or a.get('variance_weight') != 0:
        raise ValueError('D config differs beyond objective and output paths')
    return manifest, abc, d


def paired(samples, evaluations, seed=20261112, bootstrap=2000):
    if type(seed) is not int or seed < 0 or type(bootstrap) is not int or bootstrap < 100:
        raise ValueError('invalid bootstrap options')
    if set(evaluations) != {'initial','a','d'} or len(samples) != 512:
        raise ValueError('expected initial, A and D on 512 items')
    if len({x.get('id') for x in samples}) != 512:
        raise ValueError('sample IDs are not unique')
    maps = {}
    for name,e in evaluations.items():
        rows=e.get('rows')
        if not isinstance(rows,list) or len(rows)!=512: raise ValueError(name+': bad row count')
        positions={}
        for row in rows:
            i=row.get('index') if isinstance(row,dict) else None
            if type(i) is not int or not 0<=i<512 or i in positions or row.get('id')!=samples[i]['id']:
                raise ValueError(name+': bad index or ID')
            if row.get('status') not in ('ok','error','invalid_metric'):
                raise ValueError(name+': bad status')
            if row['status']=='ok' and not all(finite(row.get(k)) for k in ('falcon_si_sdri_db','streaming_si_sdri_db')):
                raise ValueError(name+': nonfinite metric')
            positions[i]=row
        valid=sum(r['status']=='ok' for r in rows)
        if e.get('summary',{}).get('attempted')!=512 or e['summary'].get('valid_pairs')!=valid or e['summary'].get('failed')!=512-valid:
            raise ValueError(name+': inconsistent summary')
        maps[name]=positions
    valid={i for i,r in maps['a'].items() if r['status']=='ok'}
    if any({i for i,r in maps[n].items() if r['status']=='ok'}!=valid for n in ('initial','d')):
        raise ValueError('valid sets differ')
    included=[]; invalid=[]; excluded=[]; max_spread=0.0
    for i,sample in enumerate(samples):
        r={name:m[i] for name,m in maps.items()}
        if i not in valid:
            invalid.append({'index':i,'id':sample['id'],'results':{n:{'status':x['status'],'error':x.get('error')} for n,x in r.items()}}); continue
        base=[x['falcon_si_sdri_db'] for x in r.values()]
        max_spread=max(max_spread,max(base)-min(base))
        if max(base)-min(base)>0.001: raise ValueError('baseline scores differ for sample '+str(i))
        group,source=sample.get('source_group_sha256'),sample.get('target_source_sha256')
        if not valid_hash(group) or not valid_hash(source):
            excluded.append({'index':i,'id':sample['id'],'reason':'missing video or target source hash'}); continue
        included.append({'index':i,'id':sample['id'],'video_group_sha256':group,
                         'falcon_si_sdri_db':r['a']['falcon_si_sdri_db'],
                         'scores':{n:x['streaming_si_sdri_db'] for n,x in r.items()}})
    groups=sorted({r['video_group_sha256'] for r in included})
    if len(groups)<2: raise ValueError('need at least two video groups')
    group_idx={g:i for i,g in enumerate(groups)}
    counts=np.zeros(len(groups),dtype=np.int64)
    specs=(('d','a'),('d','initial'),('a','initial'))
    sums={f'{a}_minus_{b}':np.zeros(len(groups)) for a,b in specs}
    for row in included:
        i=group_idx[row['video_group_sha256']]; counts[i]+=1
        for a,b in specs: sums[f'{a}_minus_{b}'][i]+=row['scores'][a]-row['scores'][b]
    draws=np.random.default_rng(seed).integers(0,len(groups),size=(bootstrap,len(groups)))
    pairs={}
    for a,b in specs:
        name=f'{a}_minus_{b}'
        delta=[r['scores'][a]-r['scores'][b] for r in included]
        distribution=sums[name][draws].sum(axis=1)/counts[draws].sum(axis=1)
        pairs[name]={'mean_db':float(np.mean(delta)),'median_db':float(np.median(delta)),
                     'bootstrap_95_ci_db':[float(x) for x in np.quantile(distribution,[.025,.975])],
                     'numerator_better_n':sum(x>0 for x in delta)}
    return {'counts':{'indexed':512,'valid_before_group_check':len(valid),'invalid':len(invalid),
                      'excluded_missing_group':len(excluded),'paired':len(included),'video_groups':len(groups)},
            'bootstrap':{'unit':'target video directory hash','seed':seed,'replicates':bootstrap,
                         'interval':'percentile 2.5/97.5','weighting':'resample groups, average selected samples'},
            'means_si_sdri_db':{n:float(np.mean([r['scores'][n] for r in included])) for n in ('initial','a','d')},
            'baseline_max_score_spread_db':max_spread,'pairs':pairs,'invalid':invalid,'excluded':excluded,'rows':included}


def compare(output_dir, evaluations, seed=20261112, bootstrap=2000):
    root=Path(output_dir).resolve(); manifest_sha=sha256(root/'manifest.json')
    manifest,abc,d=verify_d(root)
    paths={k:Path(v).resolve() for k,v in evaluations.items()}
    if set(paths)!={'initial','a','d'}: raise ValueError('need initial, A and D evaluations')
    hashes={k:sha256(v) for k,v in paths.items()}
    evals={k:json.loads(v.read_text()) for k,v in paths.items()}
    index=json.loads(Path(abc['inputs']['valid_index']['path']).read_text()); samples=index['samples']
    reference=None; baseline_path=None
    repo=Path(__file__).resolve().parents[2]
    expected_code={}
    for absolute,digest in abc['code_sha256'].items():
        try: rel=str(Path(absolute).relative_to(repo))
        except ValueError: continue
        if rel.startswith(('look2hear/models/','look2hear/videomodels/','look2hear/layers/','look2hear/losses/','look2hear/metrics/','look2hear/utils/')) or rel in ('tools/streaming/evaluate_realssa.py','tools/streaming/realssa_tar.py','tools/streaming/common.py','configs/Falcon.yml'):
            expected_code[rel]=digest
    for name,e in evals.items():
        if any(k not in e for k in PROTOCOL_KEYS): raise ValueError(name+': missing protocol')
        protocol={k:e[k] for k in PROTOCOL_KEYS}
        if reference is None: reference=protocol
        elif protocol!=reference: raise ValueError(name+': evaluation protocol differs')
        if (e['index_sha256'] != abc['inputs']['valid_index']['sha256'] or e['split']!='cv'
            or e['source_split']!=index.get('source_split') or e['partition_kind']!=index.get('partition_kind')
            or e['selection']!=index.get('selection') or e['source_provenance_basis']!='explicit'
            or e['code_sha256']!=expected_code or e.get('start')!=0 or e.get('end_exclusive')!=512):
            raise ValueError(name+': index, provenance, code or coverage mismatch')
        path=Path(e['baseline_checkpoint']).resolve()
        if baseline_path is None: baseline_path=path
        if path!=baseline_path or sha256(path)!=e['baseline_sha256']: raise ValueError(name+': baseline mismatch')
    models={}
    models['a']=_checkpoint(A,abc['variants'][A],abc,evals['a'])
    dentry={'config':str(root/'d_reconstruction.yml'),'sha256':manifest['d_config']['sha256'],'visual_mode':'causal'}
    models['d']=_checkpoint('d_reconstruction',dentry,abc,evals['d'])
    for name in ('a','d'):
        checkpoint=read_checkpoint(models[name]['path'],'cpu')
        positive=name=='a'
        if (checkpoint.get('speaker_contrastive_loss') is not positive or
            checkpoint.get('training_objective') != ('negative_si_snr_plus_positive_contrastive_no_negatives' if positive else 'negative_si_snr_reconstruction_only') or
            (checkpoint.get('speaker_pool_state') is None) == positive or
            checkpoint.get('warm_start_metadata',{}).get('speaker_pool_loaded') is not positive):
            raise ValueError(name+': saved objective/pool does not match control')
    warm=abc['inputs']['warm_start']; initial=evals['initial']
    initial_path=Path(warm['path']).resolve()
    if (Path(initial.get('streaming_checkpoint','')).resolve()!=initial_path or initial.get('streaming_sha256')!=warm['sha256'] or sha256(initial_path)!=warm['sha256']):
        raise ValueError('initial evaluation is not the frozen warm-start checkpoint')
    ckpt=read_checkpoint(initial_path,'cpu')
    if initial.get('streaming_steps')!=ckpt.get('steps') or initial.get('streaming_model_config')!=ckpt['model_config']:
        raise ValueError('initial checkpoint metadata differs')
    result=paired(samples,evals,seed,bootstrap)
    if any(sha256(p)!=hashes[n] for n,p in paths.items()) or sha256(baseline_path)!=reference['baseline_sha256'] or sha256(initial_path)!=warm['sha256']:
        raise RuntimeError('evaluation or checkpoint changed during comparison')
    verify_d(root)
    if sha256(root/'manifest.json')!=manifest_sha: raise RuntimeError('D manifest changed')
    result.update(format='falcon-quality-d-paired-v1',scope='paired central-2s ch0 on derived tr development; warm-start saw original tr',
                  caveat='target video directories, not verified speakers; one training seed; initial and trained checkpoints differ in step count',
                  manifest_sha256=manifest_sha,abc_manifest_sha256=manifest['abc_manifest_sha256'],
                  comparison_code_sha256=sha256(Path(__file__).resolve()), numpy_version=str(np.__version__),
                  models=models,initial_checkpoint={'path':str(initial_path),'sha256':warm['sha256'],'steps':ckpt.get('steps')},
                  evaluations={n:{'path':str(p),'sha256':hashes[n]} for n,p in paths.items()},
                  input_index_sha256=abc['inputs']['valid_index']['sha256'],baseline_sha256=reference['baseline_sha256'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    prepare_p=sub.add_parser('prepare'); prepare_p.add_argument('--abc-dir',required=True);prepare_p.add_argument('--output-dir',required=True)
    verify_p=sub.add_parser('verify'); verify_p.add_argument('--output-dir',required=True)
    cmp_p=sub.add_parser('compare');cmp_p.add_argument('--output-dir',required=True)
    for key in ('initial','a','d'): cmp_p.add_argument('--'+key,required=True)
    cmp_p.add_argument('--output',required=True);cmp_p.add_argument('--seed',type=int,default=20261112);cmp_p.add_argument('--bootstrap',type=int,default=2000)
    a=p.parse_args()
    if a.command=='prepare': result=prepare(a.abc_dir,a.output_dir)
    elif a.command=='verify': result={'verified':True,'manifest':verify_d(a.output_dir)[0]}
    else:
        out=Path(a.output)
        if out.exists(): raise FileExistsError(out)
        result=compare(a.output_dir,{k:getattr(a,k) for k in ('initial','a','d')},a.seed,a.bootstrap)
        out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('x') as f: json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
        result={'counts':result['counts'],'pairs':result['pairs'],'output':str(out)}
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__': main()
