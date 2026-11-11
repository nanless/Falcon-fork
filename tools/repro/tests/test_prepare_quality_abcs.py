import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from tools.streaming.prepare_quality_abcs import prepare, verify


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def save_index(path, role, samples, **extra):
    value={'format':'realssa-tar-index-v1','split':role,'source_split':'tr',
           'partition_kind':'derived_video_directory_group_holdout',
           'sample_count':len(samples),'samples':samples, **extra}
    path.write_text(json.dumps(value))
    return value


def fixture_paths(tmp_path):
    visual=tmp_path/'visual.pt';visual.write_bytes(b'visual')
    base=tmp_path/'base.yml';base.write_text(yaml.safe_dump({'visual_init':str(visual),
        'model_channels':320,'epochs':300,'backend':'nccl','workers':4,'positive_contrastive':True}))
    train=tmp_path/'tr.json'
    save_index(train,'tr',[{'id':'train','source_group_sha256':digest('train-video'),
                            'target_source_sha256':digest('train-target')}])
    parent=tmp_path/'cv.json'
    cv=[{'id':f'cv-{i}','source_group_sha256':digest('cv-video-'+str(i//2)),
         'target_source_sha256':digest('cv-target-'+str(i))} for i in range(512)]
    save_index(parent,'cv',cv)
    parent_sha=hashlib.sha256(parent.read_bytes()).hexdigest()
    fast=tmp_path/'cv_fast.json'
    save_index(fast,'cv',cv,selection={'role':'fast_dev_from_source_tr','parent_index_sha256':parent_sha})
    source_sha=digest('original-full-tr')
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'format':'falcon-realssa-source-holdout-v1',
        'source':{'index_sha256':source_sha},
        'derived_index_sha256':{'tr':hashlib.sha256(train.read_bytes()).hexdigest(),
                                'cv':parent_sha,'tt':digest('unused')}}))
    ckpt=tmp_path/'model.pt'
    torch.save({'format':'falcon-streaming-v4',
                'state_dict':{'_streaming_format_version':torch.tensor(4,dtype=torch.int64)},
                'model_config':{'channels':320,'dilations':[1,2,4,8,16,32]},
                'speaker_pool_state':{},'visual_initialization_sha256':hashlib.sha256(visual.read_bytes()).hexdigest(),
                'train_tar_index_sha256':source_sha,'steps':4900},ckpt)
    return base,train,fast,ckpt,manifest


def test_configs_vary_only_intended_fields_and_never_launch(tmp_path):
    args=fixture_paths(tmp_path);out=tmp_path/'experiment'
    manifest=prepare(*args,out,max_steps=50,lr=6.25e-5,warmup_steps=0)
    configs={name:yaml.safe_load(Path(v['config']).read_text()) for name,v in manifest['variants'].items()}
    assert all(c['batch_size']==1024 and c['accum']==1 and c['max_steps']==50 and c['lr']==6.25e-5 for c in configs.values())
    assert all(c['exclude_degenerate_contrastive'] and c['silence_rms_threshold']==0 and c['warmup_steps']==0 for c in configs.values())
    variants={name:(c['visual_mode'],c['variance_weight']) for name,c in configs.items()}
    assert variants=={'a_causal_positive':('causal',0.0),'b_causal_variance':('causal',.1),'c_centered_variance':('chunk_centered',.1)}
    omit={'visual_mode','variance_weight','output','checkpoint_dir'}
    assert len({json.dumps({k:v for k,v in c.items() if k not in omit},sort_keys=True) for c in configs.values()})==1
    assert all(not Path(c['output']).exists() and not Path(c['checkpoint_dir']).exists() for c in configs.values())
    assert not (tmp_path/'tt.json').exists()
    assert manifest['data_roles']['holdout_tt_not_opened_or_evaluated']
    assert 'not unseen-source' in manifest['warning']
    assert verify(out)['format']=='falcon-quality-abcs-v1'


def test_no_overwrite_and_detect_changed_config(tmp_path):
    args=fixture_paths(tmp_path);out=tmp_path/'experiment'
    manifest=prepare(*args,out,max_steps=2)
    with pytest.raises(FileExistsError):prepare(*args,out,max_steps=2)
    Path(manifest['variants']['a_causal_positive']['config']).write_text('tampered: true\n')
    with pytest.raises(ValueError,match='prepared config changed'):verify(out)


def test_rejects_fast_subset_not_identical_to_parent(tmp_path):
    base,train,fast,ckpt,manifest=fixture_paths(tmp_path)
    data=json.loads(fast.read_text());data['samples'][0]['target_source_sha256']=digest('different')
    fast.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='rows differ'):
        prepare(base,train,fast,ckpt,manifest,tmp_path/'experiment')
    assert not (tmp_path/'experiment').exists()


def test_rejects_warmstart_not_trained_on_original_full_tr(tmp_path):
    base,train,fast,ckpt,manifest=fixture_paths(tmp_path)
    payload=torch.load(ckpt,weights_only=True);payload['train_tar_index_sha256']=digest('other')
    torch.save(payload,ckpt)
    with pytest.raises(ValueError,match='not the expected'):
        prepare(base,train,fast,ckpt,manifest,tmp_path/'experiment')


@pytest.mark.parametrize('keyword,value',[('max_steps',0),('lr',0),('variance_weight',0),('warmup_steps',-1)])
def test_rejects_invalid_settings_before_creating_output(tmp_path,keyword,value):
    args=fixture_paths(tmp_path);out=tmp_path/'experiment'
    with pytest.raises(ValueError):prepare(*args,out,**{keyword:value})
    assert not out.exists()
