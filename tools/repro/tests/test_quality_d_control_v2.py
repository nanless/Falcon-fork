import copy
import json
from pathlib import Path
import pytest
import yaml
from tools.streaming import quality_d_control_v2 as d


def samples():
    return [{'id':f'id-{i}','source_group_sha256':format(i%20,'064x'),
             'target_source_sha256':format(i+1,'064x')} for i in range(512)]


def evaluations(items):
    out={}
    for name,score in [('initial',0.),('a',1.),('d',3.)]:
        out[name]={'summary':{'attempted':512,'valid_pairs':512,'failed':0},
                   'rows':[{'id':x['id'],'index':i,'status':'ok','falcon_si_sdri_db':4.,
                            'streaming_si_sdri_db':score+i/100} for i,x in enumerate(items)]}
    return out


def test_paired_differences_and_reproducible_cluster_ci():
    s=samples(); e=evaluations(s)
    r=d.paired(s,e,seed=42,bootstrap=200)
    assert r == d.paired(s,e,seed=42,bootstrap=200)
    assert r['counts']['paired']==512 and r['counts']['video_groups']==20
    assert r['pairs']['d_minus_a']['mean_db']==pytest.approx(2)
    assert r['pairs']['d_minus_a']['bootstrap_95_ci_db']==pytest.approx([2,2])
    assert r['pairs']['d_minus_initial']['mean_db']==pytest.approx(3)


def test_rejects_different_valid_set_duplicate_identity_and_baseline():
    s=samples(); e=evaluations(s)
    e['d']['rows'][0]['status']='error';e['d']['summary'].update(valid_pairs=511,failed=1)
    with pytest.raises(ValueError,match='valid sets differ'): d.paired(s,e)
    e=evaluations(s);e['a']['rows'][0]['id']='wrong'
    with pytest.raises(ValueError,match='bad index or ID'): d.paired(s,e)
    e=evaluations(s);e['d']['rows'][0]['falcon_si_sdri_db']+=1
    with pytest.raises(ValueError,match='baseline scores differ'): d.paired(s,e)


def test_common_failure_and_missing_source_are_audited():
    s=samples();s[1]['source_group_sha256']=None;e=evaluations(s)
    for x in e.values():
        x['rows'][0]['status']='error';x['summary'].update(valid_pairs=511,failed=1)
    r=d.paired(s,e,bootstrap=100)
    assert (r['counts']['invalid'],r['counts']['excluded_missing_group'],r['counts']['paired'])==(1,1,510)
    assert r['invalid'][0]['id']=='id-0' and r['excluded'][0]['id']=='id-1'


def test_prepare_only_changes_requested_factor_and_output_and_rejects_mutation(tmp_path,monkeypatch):
    abc=tmp_path/'abc';abc.mkdir();(abc/'manifest.json').write_text('{}')
    a={'positive_contrastive':True,'variance_weight':0.,'visual_mode':'causal','max_steps':50,
       'warm_start':'/same','seed':42,'tar_index':'/train','valid_tar_index':'/valid',
       'output':'/old/final.pt','checkpoint_dir':'/old/epochs'}
    ap=abc/'a.yml';ap.write_text(yaml.safe_dump(a))
    m={'variants':{d.A:{'config':str(ap),'sha256':d.sha256(ap),'visual_mode':'causal','variance_weight':0.}}}
    monkeypatch.setattr(d,'verify_abc',lambda path:m)
    out=tmp_path/'d'; manifest=d.prepare(abc,out)
    got=yaml.safe_load((out/'d_reconstruction.yml').read_text())
    assert got==dict(a,positive_contrastive=False,reconstruction_only=True,output=str(out/'final.pt'),checkpoint_dir=str(out/'epochs'))
    assert d.verify_d(out)[0]==manifest
    with pytest.raises(FileExistsError):d.prepare(abc,out)
    got['seed']=7;(out/'d_reconstruction.yml').write_text(yaml.safe_dump(got))
    with pytest.raises(ValueError,match='D config changed'):d.verify_d(out)


def test_compare_audits_provenance_initial_identity_and_objective(tmp_path,monkeypatch):
    root=tmp_path/'d';root.mkdir();(root/'manifest.json').write_text('{}')
    s=samples();index=tmp_path/'index.json'
    index.write_text(json.dumps({'samples':s,'source_split':'tr','partition_kind':'derived_video_directory_group_holdout','selection':{'seed':42}}))
    warm=tmp_path/'warm.pt';warm.write_bytes(b'warm')
    baseline=tmp_path/'baseline.pt';baseline.write_bytes(b'base')
    abc={'inputs':{'valid_index':{'path':str(index),'sha256':d.sha256(index)},
                   'warm_start':{'path':str(warm),'sha256':d.sha256(warm)}},
         'variants':{d.A:{}},'code_sha256':{}}
    manifest={'abc_manifest_sha256':'abc','d_config':{'sha256':'config'}}
    monkeypatch.setattr(d,'verify_d',lambda _: (manifest,abc,{}))
    monkeypatch.setattr(d,'_checkpoint',lambda name,entry,manifest,e: {'path':name+'.pt','sha256':name})
    def ckpt(path,*_):
        if Path(path)==warm:return {'steps':4900,'model_config':{'channels':320}}
        positive=Path(path).name==d.A+'.pt'
        return {'speaker_contrastive_loss':positive,
                'training_objective':'negative_si_snr_plus_positive_contrastive_no_negatives' if positive else 'negative_si_snr_reconstruction_only',
                'speaker_pool_state':{} if positive else None,'warm_start_metadata':{'speaker_pool_loaded':positive}}
    monkeypatch.setattr(d,'read_checkpoint',ckpt)
    ev=evaluations(s); files={}
    for name,e in ev.items():
        for k in d.PROTOCOL_KEYS: e[k]='same'
        e.update(index_sha256=d.sha256(index),split='cv',source_split='tr',
                 partition_kind='derived_video_directory_group_holdout',selection={'seed':42},
                 source_provenance_basis='explicit',code_sha256={},start=0,end_exclusive=512,
                 baseline_checkpoint=str(baseline),baseline_sha256=d.sha256(baseline))
        if name=='initial': e.update(streaming_checkpoint=str(warm),streaming_sha256=d.sha256(warm),streaming_steps=4900,streaming_model_config={'channels':320})
        p=tmp_path/(name+'.json');p.write_text(json.dumps(e));files[name]=p
    result=d.compare(root,files,bootstrap=100)
    assert result['pairs']['d_minus_a']['mean_db']==pytest.approx(2)
    ev['initial']['streaming_sha256']='0'*64;files['initial'].write_text(json.dumps(ev['initial']))
    with pytest.raises(ValueError,match='initial evaluation'): d.compare(root,files,bootstrap=100)
    ev['initial']['streaming_sha256']=d.sha256(warm);files['initial'].write_text(json.dumps(ev['initial']))
    ev['d']['partition_kind']='official';files['d'].write_text(json.dumps(ev['d']))
    with pytest.raises(ValueError,match='protocol differs'): d.compare(root,files,bootstrap=100)


def test_real_trainer_parser_accepts_reconstruction_objective(tmp_path,monkeypatch):
    from tools.streaming.train_ddp import parse_args
    abc=tmp_path/'abc';abc.mkdir();(abc/'manifest.json').write_text('{}')
    a={'positive_contrastive':True,'variance_weight':0.,'visual_mode':'causal','max_steps':50,
       'warm_start':'/same','seed':42,'tar_index':'/train','valid_tar_index':'/valid',
       'visual_init':'/visual','output':'/old/final.pt','checkpoint_dir':'/old/epochs'}
    ap=abc/'a.yml';ap.write_text(yaml.safe_dump(a))
    m={'variants':{d.A:{'config':str(ap),'sha256':d.sha256(ap),'visual_mode':'causal','variance_weight':0.}}}
    monkeypatch.setattr(d,'verify_abc',lambda _:m)
    out=tmp_path/'d';d.prepare(abc,out)
    args=parse_args(['--config',str(out/'d_reconstruction.yml')])
    assert args.reconstruction_only is True and args.positive_contrastive is False
    assert args.max_steps==50 and args.variance_weight==0
