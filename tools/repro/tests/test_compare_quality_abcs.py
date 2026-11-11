import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

from tools.repro.tests.test_prepare_quality_abcs import fixture_paths
from tools.streaming.compare_quality_abcs import compare, main
from tools.streaming.prepare_quality_abcs import prepare, sha256


def evaluations(tmp_path):
    args=fixture_paths(tmp_path)
    exp=tmp_path/'exp'
    manifest=prepare(*args,exp,max_steps=50)
    index=json.loads(Path(manifest['inputs']['valid_index']['path']).read_text())
    baseline=tmp_path/'baseline.pt';baseline.write_bytes(b'same-open-falcon')
    repo=Path(__file__).resolve().parents[3]
    code={}
    for path,digest in manifest['code_sha256'].items():
        rel=str(Path(path).relative_to(repo))
        if (rel.startswith(('look2hear/models/','look2hear/videomodels/','look2hear/layers/',
                            'look2hear/losses/','look2hear/metrics/','look2hear/utils/'))
            or rel in ('tools/streaming/evaluate_realssa.py','tools/streaming/realssa_tar.py',
                       'tools/streaming/common.py','configs/Falcon.yml')):
            code[rel]=digest
    names=list(manifest['variants'])
    offsets=[0.0,1.0,3.0]
    paths={}
    for name,offset in zip(names,offsets):
        entry=manifest['variants'][name]
        config=yaml.safe_load(Path(entry['config']).read_text())
        model_config={'channels':320,'dilations':[1,2,4,8,16,32]}
        if entry['visual_mode']!='causal':model_config['visual_mode']=entry['visual_mode']
        version=4 if entry['visual_mode']=='causal' else 5
        final=Path(config['output']);final.parent.mkdir()
        protocol={}
        for key,rel in (('trainer_code_sha256','tools/streaming/train_ddp.py'),
                        ('config_parser_sha256','tools/streaming/training_config.py'),
                        ('data_code_sha256','tools/streaming/realssa_tar.py')):
            protocol[key]=manifest['code_sha256'][str((repo/rel).resolve())]
        ckpt={'format':'falcon-streaming-v'+str(version),'model_config':model_config,
              'state_dict':{'_streaming_format_version':torch.tensor(version,dtype=torch.int64)},
              'steps':50,'configuration_sha256':entry['sha256'],'effective_arguments':dict(config),
              'warm_start_metadata':{'sha256':manifest['inputs']['warm_start']['sha256'],'optimizer_scheduler_and_steps_reset':True},
              'train_tar_index_sha256':manifest['inputs']['train_index']['sha256'],
              'validation_tar_index_sha256':manifest['inputs']['valid_index']['sha256'],
              'visual_initialization_sha256':manifest['inputs']['visual_init']['sha256'],
              'training_status':'real_data_optimization','world_size':2,'batch_per_rank':1024,'gradient_accumulation':1,
              'target_learning_rate':config['lr'],'model_code_sha256':manifest['code_sha256'][str((repo/'look2hear/models/FalconStreaming.py').resolve())],
              'resume_protocol':protocol}
        for key in ('variance_weight','variance_floor','silence_rms_threshold','silence_loss_weight','exclude_degenerate_contrastive'):
            ckpt[key]=config[key]
        torch.save(ckpt,final)
        rows=[]
        for i,sample in enumerate(index['samples']):
            if i==5:
                rows.append({'id':sample['id'],'index':i,'status':'error','error':'zero target'})
            else:
                rows.append({'id':sample['id'],'index':i,'status':'ok','falcon_si_sdri_db':2.0+i/1000,
                             'streaming_si_sdri_db':i/1000+offset})
        evaluation={'scope':'same center test','split':'cv','source_split':'tr','partition_kind':index['partition_kind'],
                    'selection':index['selection'],'source_provenance_basis':'explicit','index_sha256':manifest['inputs']['valid_index']['sha256'],
                    'baseline_checkpoint':str(baseline),'baseline_sha256':sha256(baseline),'baseline_epoch':714,
                    'preprocessing':'same','metric':'SI-SDRi','short_input_padding':'same','code_hash_scope':'same',
                    'code_sha256':code,'torch_version':'x','device':'cuda','threads':2,'start':0,'end_exclusive':512,
                    'streaming_checkpoint':str(final),'streaming_sha256':sha256(final),'streaming_steps':50,
                    'streaming_model_config':model_config,'summary':{'attempted':512,'valid_pairs':511,'failed':1},'rows':rows}
        p=tmp_path/(name+'.json');p.write_text(json.dumps(evaluation));paths[name]=p
    return exp,paths


def test_paired_comparison_uses_clusters_and_records_invalid(tmp_path):
    exp,paths=evaluations(tmp_path)
    result=compare(exp,paths,seed=7,bootstrap=200)
    repeated=compare(exp,paths,seed=7,bootstrap=200)
    assert result==repeated
    assert result['counts']=={'indexed':512,'valid_before_group_check':511,'invalid':1,'excluded_missing_group':0,'paired':511,'video_groups':256}
    assert result['invalid'][0]['index']==5
    assert result['pairs']['b_causal_variance_minus_a_causal_positive']['mean_db']==pytest.approx(1)
    assert result['pairs']['b_causal_variance_minus_a_causal_positive']['bootstrap_95_ci_db']==pytest.approx([1,1])
    assert result['pairs']['c_centered_variance_minus_b_causal_variance']['mean_db']==pytest.approx(2)
    assert 'not verified speakers' in result['source_note']
    assert result['comparison_code_sha256']==sha256(result['comparison_code'])
    assert result['numpy_version']


def test_rejects_mismatched_row_id(tmp_path):
    exp,paths=evaluations(tmp_path);p=paths['b_causal_variance']
    d=json.loads(p.read_text());d['rows'][1]['id']='wrong';p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='sample ID mismatch'):compare(exp,paths,bootstrap=100)


def test_rejects_different_valid_sets(tmp_path):
    exp,paths=evaluations(tmp_path);p=paths['b_causal_variance']
    d=json.loads(p.read_text());d['rows'][8]['status']='error';d['summary']['valid_pairs']-=1;d['summary']['failed']+=1;p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='valid-sample set differs'):compare(exp,paths,bootstrap=100)


def test_rejects_baseline_protocol_mismatch(tmp_path):
    exp,paths=evaluations(tmp_path);p=paths['b_causal_variance']
    d=json.loads(p.read_text());d['baseline_sha256']='bad';p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='protocol differs'):compare(exp,paths,bootstrap=100)


def test_rejects_changed_checkpoint(tmp_path):
    exp,paths=evaluations(tmp_path);d=json.loads(paths['a_causal_positive'].read_text())
    Path(d['streaming_checkpoint']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint digest mismatch'):compare(exp,paths,bootstrap=100)


def test_rejects_effective_override_even_if_config_hash_matches(tmp_path):
    exp,paths=evaluations(tmp_path);p=paths['a_causal_positive'];d=json.loads(p.read_text())
    model=Path(d['streaming_checkpoint']);ckpt=torch.load(model,weights_only=True)
    ckpt['effective_arguments']['lr']=0.01;torch.save(ckpt,model)
    d['streaming_sha256']=sha256(model);p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='effective option differs.*lr'):compare(exp,paths,bootstrap=100)


def test_rejects_source_or_eval_code_change(tmp_path):
    exp,paths=evaluations(tmp_path);p=paths['a_causal_positive']
    d=json.loads(p.read_text());d['code_sha256']={};p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='index or evaluation code digest mismatch'):compare(exp,paths,bootstrap=100)


def test_rejects_existing_output_before_reading_inputs(tmp_path,monkeypatch):
    out=tmp_path/'exists.json';out.write_text('preserve')
    monkeypatch.setattr(sys,'argv',['compare','--experiment-dir','absent','--a','a','--b','b','--c','c','--output',str(out)])
    with pytest.raises(FileExistsError):main()
    assert out.read_text()=='preserve'


def test_missing_source_hash_is_excluded_and_audited_consistently(tmp_path):
    # Simulate an otherwise self-consistent prior manifest whose single dev row
    # has unknown grouping; regular prepare currently rejects such input earlier.
    exp,paths=evaluations(tmp_path)
    manifest_path=exp/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    index_path=Path(manifest['inputs']['valid_index']['path'])
    data=json.loads(index_path.read_text())
    data['samples'][0]['source_group_sha256']=None
    index_path.write_text(json.dumps(data))
    revised=sha256(index_path)
    manifest['inputs']['valid_index']['sha256']=revised
    manifest_path.write_text(json.dumps(manifest))
    for path in paths.values():
        evaluation=json.loads(path.read_text())
        model=Path(evaluation['streaming_checkpoint'])
        checkpoint=torch.load(model,weights_only=True)
        checkpoint['validation_tar_index_sha256']=revised
        torch.save(checkpoint,model)
        evaluation['index_sha256']=revised
        evaluation['streaming_sha256']=sha256(model)
        path.write_text(json.dumps(evaluation))
    result=compare(exp,paths,seed=7,bootstrap=100)
    assert result['counts']['invalid']==1
    assert result['counts']['excluded_missing_group']==1
    assert result['counts']['paired']==510
    assert result['excluded']==[{'index':0,'id':'cv-0','reason':'missing_or_invalid_video_or_target_source_hash'}]
    assert result['pairs']['b_causal_variance_minus_a_causal_positive']['mean_db']==pytest.approx(1)


def test_rejects_consistent_but_forged_original_source_role(tmp_path):
    exp,paths=evaluations(tmp_path)
    for path in paths.values():
        d=json.loads(path.read_text())
        d['source_split']='cv'
        path.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='derived development provenance mismatch'):
        compare(exp,paths,bootstrap=100)
