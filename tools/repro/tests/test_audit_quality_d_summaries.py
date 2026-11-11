import pytest
from tools.streaming.audit_quality_d_summaries import audit_eval,audit_train


def test_eval_summary_recomputed_and_mismatch_rejected():
    rows=[{'status':'ok','falcon_si_sdri_db':float(i),'streaming_si_sdri_db':float(i)+2} for i in range(512)]
    e={'rows':rows,'summary':{'attempted':512,'valid_pairs':512,'failed':0,
       'falcon_si_sdri_db_mean':255.5,'streaming_si_sdri_db_mean':257.5,'streaming_minus_falcon_db':2.}}
    assert audit_eval(e)['streaming_si_sdri_db']==257.5
    e['summary']['streaming_si_sdri_db_mean']+=1
    with pytest.raises(ValueError,match='mean differs'): audit_eval(e)


def fixture():
    c={'steps':50,'end_reason':'max_steps','validation_samples':512,'validation_total':512,
       'validation_limit':0,'world_size':2,'batch_per_rank':1024,'gradient_accumulation':1,
       'resume_restarts_epoch':True,'next_epoch':1,'target_learning_rate':6.25e-5,
       'learning_rate':6.25e-5,'validation_loss':-5.1}
    first=[f'epoch=0 step={i} loss_last_microbatch=-5' for i in range(1,40)]
    second=[f'epoch=1 step={i} loss_last_microbatch=-5' for i in range(40,51)]
    log='\n'.join(first+['epoch=0 validation_samples=512 validation_total=512 validation_limit=0 loss=-5.0 lr=6.25e-05 partial=False']+second+
                  ['epoch=1 validation_samples=512 validation_total=512 validation_limit=0 loss=-5.100000 lr=6.25e-05 partial=True','saved /some/final.pt'])
    return c,log


def test_training_summary_matches_steps_and_loss():
    c,log=fixture();assert audit_train(c,log,'0')['steps']==50
    with pytest.raises(ValueError,match='exit status'): audit_train(c,log,'1')
    with pytest.raises(ValueError,match='unexpected steps'): audit_train(c,log.replace('epoch=1 step=40 ','epoch=1 step=41 '),'0')
    c['validation_loss']=-5.2
    with pytest.raises(ValueError,match='loss differs'): audit_train(c,log,'0')
