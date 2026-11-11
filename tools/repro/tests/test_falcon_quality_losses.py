import copy
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

from look2hear.losses import pairwise_neg_sisdr
from look2hear.losses.nass_contrastive_loss import NASSContrastiveLoss
from look2hear.models.FalconStreaming import FalconStreaming
from tools.streaming.train_ddp import (
    TrainingWrapper, reconstruction_components, advance_validation_dual,
    load_warm_start, parse_args, validate_resume_protocol,
)


def test_hybrid_preserves_old_speech_loss_and_gradient_exactly():
    gen=torch.Generator().manual_seed(72)
    target=torch.randn(3,1024,generator=gen)
    mixture=torch.randn(3,1024,generator=gen)
    output=torch.randn(3,1024,generator=gen,requires_grad=True)
    combined, old, nmse, degenerate, silent=reconstruction_components(output,target,mixture)
    assert not degenerate.any() and not silent.any()
    expected=pairwise_neg_sisdr(output[:,None],target[:,None]).reshape(-1)
    torch.testing.assert_close(combined,expected,rtol=0,atol=0)
    gradient=torch.autograd.grad(combined.mean(),output,retain_graph=True)[0]
    expected_gradient=torch.autograd.grad(expected.mean(),output)[0]
    torch.testing.assert_close(gradient,expected_gradient,rtol=0,atol=0)


def test_hybrid_trains_silence_and_constant_target_in_mixed_batch():
    target=torch.tensor([[0.,0.,0.,0.],[.1,.1,.1,.1],[.1,-.1,.2,-.2]])
    mixture=torch.ones_like(target)
    output=torch.full_like(target,.25,requires_grad=True)
    combined,sisdr,nmse,degenerate,silent=reconstruction_components(output,target,mixture)
    assert degenerate.tolist()==[True,True,False] and silent.tolist()==[True,False,False]
    assert combined[0].item()==pytest.approx(.0625)
    assert combined[1].item()==pytest.approx(.0225)
    combined.sum().backward()
    assert torch.count_nonzero(output.grad[0])==4
    assert torch.count_nonzero(output.grad[1])==4
    assert torch.all(output.grad[0]>0) and torch.isfinite(output.grad).all()


def test_hybrid_threshold_is_explicit_and_zero_mix_is_finite():
    target=torch.tensor([[1e-6,-1e-6,1e-6,-1e-6]])
    mixture=torch.zeros_like(target)
    output=torch.full_like(target,.01,requires_grad=True)
    assert not reconstruction_components(output,target,mixture,0.0)[3].item()
    combined,_,_,degenerate,silent=reconstruction_components(output,target,mixture,1e-5)
    assert degenerate.item() and silent.item()
    combined.sum().backward()
    assert torch.isfinite(combined).all() and torch.isfinite(output.grad).all()
    assert torch.count_nonzero(output.grad)>0


def test_vectorized_positive_matches_original_value_and_all_gradients():
    torch.manual_seed(42)
    q0=torch.randn(7,64);p=torch.randn(7,64)
    q1=q0.clone().requires_grad_();q2=q0.clone().requires_grad_()
    old=NASSContrastiveLoss(alpha_pos=.7)(q1,p,[None]*7,beta=0).squeeze()
    new=.7*(1-F.cosine_similarity(q2,p,dim=1)).mean()
    torch.testing.assert_close(old,new,rtol=1e-6,atol=1e-7)
    go=torch.autograd.grad(old,q1)[0];gn=torch.autograd.grad(new,q2)[0]
    torch.testing.assert_close(go,gn,rtol=1e-6,atol=1e-8)


def test_variance_head_only_and_fewer_than_two_active_items():
    torch.manual_seed(12)
    model=FalconStreaming(channels=8,dilations=(1,))
    wrapper=TrainingWrapper(model,True,variance_weight=1.0,variance_floor=1.0)
    target=torch.randn(3,32000,requires_grad=True)
    penalty=wrapper.variance_penalty(target)
    assert penalty.item()>0
    penalty.backward()
    assert target.grad is None
    assert all(p.grad is None for p in model.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in wrapper.speaker_pool.parameters())
    wrapper.zero_grad(set_to_none=True)
    one=torch.cat((torch.randn(1,32000),torch.zeros(2,32000)))
    loss=wrapper.variance_penalty(one)
    assert loss.item()==0 and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad)==0 for p in wrapper.speaker_pool.parameters())


def test_dual_best_tracks_true_min_while_tolerance_controls_staleness():
    class S:
        def __init__(self):self.seen=[]
        def step(self,x):self.seen.append(x)
    scheduler=S()
    numeric,significant,stale,save=advance_validation_dual(scheduler,-1.005,-1.,-1.,0,False,.01)
    assert numeric==-1.005 and significant==-1. and stale==1 and save
    numeric,significant,stale,save=advance_validation_dual(scheduler,-1.008,numeric,significant,stale,False,.01)
    assert numeric==-1.008 and significant==-1. and stale==2 and save
    numeric,significant,stale,save=advance_validation_dual(scheduler,-1.011,numeric,significant,stale,False,.01)
    assert numeric==-1.011 and significant==-1.011 and stale==0 and save
    assert scheduler.seen==[-1.005,-1.008,-1.011]
    assert advance_validation_dual(scheduler,-2.,numeric,significant,stale,True,.01)==(numeric,significant,stale,False)
    assert advance_validation_dual(scheduler,-2.,numeric,significant,stale,False,.01,policy_suppressed=True)==(-2.,significant,stale,True)
    assert scheduler.seen==[-1.005,-1.008,-1.011]


def test_warm_start_strictly_transfers_only_model_and_optional_pool(tmp_path):
    torch.manual_seed(9)
    source_model=FalconStreaming(channels=8,dilations=(1,),visual_mode='causal')
    source_wrap=TrainingWrapper(source_model,True)
    path=tmp_path/'source.pt'
    source={'format':'falcon-streaming-v4','state_dict':source_model.state_dict(),
            'model_config':{'channels':8,'dilations':[1]},'speaker_pool_state':source_wrap.speaker_pool.state_dict(),
            'visual_initialization':'random_frozen','visual_initialization_sha256':None,'steps':42,
            'optimizer':{'ignored':'fresh-start'}}
    torch.save(source,path)
    target_model=FalconStreaming(channels=8,dilations=(1,),visual_mode='chunk_centered')
    target_wrap=TrainingWrapper(target_model,True)
    meta=load_warm_start(path,target_model,target_wrap,None,random_visual=True,require_speaker_pool=True)
    assert meta['source_steps']==42 and meta['optimizer_scheduler_and_steps_reset'] is True
    assert meta['speaker_pool_loaded'] is True and len(meta['speaker_pool_sha256'])==64
    for key,value in source_model.state_dict().items():
        if key=='_streaming_format_version':
            assert target_model.state_dict()[key].item()==5
        else:torch.testing.assert_close(target_model.state_dict()[key],value,rtol=0,atol=0)
    for key,value in source_wrap.speaker_pool.state_dict().items():
        torch.testing.assert_close(target_wrap.speaker_pool.state_dict()[key],value,rtol=0,atol=0)
    with pytest.raises(ValueError,match='channels or dilations'):
        load_warm_start(path,FalconStreaming(channels=16,dilations=(1,)),target_wrap,None,
                        random_visual=True,require_speaker_pool=True)
    altered=copy.deepcopy(source);del altered['state_dict']['audio_proj.weight']
    broken=tmp_path/'broken.pt';torch.save(altered,broken)
    with pytest.raises(RuntimeError,match='Missing key'):
        load_warm_start(broken,target_model,target_wrap,None,random_visual=True,require_speaker_pool=True)


def test_init_modes_and_old_resume_protocol_are_explicit():
    args=['--synthetic','--random-visual','--positive-contrastive','--output','x']
    parsed=parse_args(args)
    assert parsed.silence_rms_threshold==0.0 and parsed.variance_weight==0.0 and parsed.visual_mode=='causal'
    with pytest.raises(SystemExit):parse_args(args+['--resume','a','--warm-start','b'])
    with pytest.raises(SystemExit):parse_args(['--synthetic','--random-visual','--reconstruction-only',
                                               '--output','x','--variance-weight','1'])
    with pytest.raises(ValueError,match='validation_tracking_version'):
        validate_resume_protocol({'train':1},{'train':1,'validation_tracking_version':2})


def test_positive_loss_can_exclude_degenerate_and_all_silent_preserves_graph():
    torch.manual_seed(88)
    wrapper=TrainingWrapper(FalconStreaming(channels=8,dilations=(1,)),True,
                            exclude_degenerate_contrastive=True)
    output=torch.randn(3,32000,requires_grad=True)
    target=torch.stack((torch.randn(32000),torch.zeros(32000),torch.full((32000,),.01)))
    mixed=wrapper.positive_loss(output,target)
    valid_only=wrapper.positive_loss(output[:1],target[:1])
    torch.testing.assert_close(mixed,valid_only,rtol=1e-6,atol=1e-7)
    zero=wrapper.positive_loss(output[1:],target[1:])
    assert zero.item()==0 and zero.requires_grad
    zero.backward()
    assert output.grad is not None and torch.count_nonzero(output.grad)==0
    assert all(p.grad is not None for p in wrapper.speaker_pool.parameters())
    assert all(torch.count_nonzero(p.grad)==0 for p in wrapper.speaker_pool.parameters())


def test_positive_mask_is_opt_in_for_reproducible_baseline():
    torch.manual_seed(91)
    wrapper=TrainingWrapper(FalconStreaming(channels=8,dilations=(1,)),True)
    out=torch.randn(2,32000);target=torch.stack((torch.randn(32000),torch.zeros(32000)))
    q=wrapper.speaker_pool(wrapper.encode_waveform(out))
    with torch.no_grad():p=wrapper.speaker_pool(wrapper.encode_waveform(target))
    expected=NASSContrastiveLoss()(q,p,[None]*2,beta=0).squeeze()
    torch.testing.assert_close(wrapper.positive_loss(out,target),expected,rtol=1e-6,atol=1e-7)


def _distributed_positive_worker(rank, path, outdir, masks):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from tools.streaming.train_ddp import masked_positive_cosine_loss, reduce_training_loss_metrics
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+str(path),rank=rank,world_size=2)
    try:
        torch.manual_seed(412)
        layer=torch.nn.Linear(3,4,bias=False)
        ddp=DistributedDataParallel(layer)
        inputs=torch.tensor([[.1,.2,.3],[.4,-.5,.6],[.3,.9,-.7],[.2,.1,-.8]])[rank*2:rank*2+2]
        positives=torch.tensor([[.7,.2,.1,-.4],[.3,-.1,.4,.5],[.2,.7,-.3,.2],[-.5,.1,.9,.2]])[rank*2:rank*2+2]
        loss=masked_positive_cosine_loss(ddp(inputs),positives,torch.tensor(masks[rank],dtype=torch.bool))
        positive_global, variance_mean = reduce_training_loss_metrics(loss,torch.tensor(float(rank+1)),torch.device('cpu'))
        loss.backward()
        torch.save({'gradient':layer.weight.grad,'positive_global':positive_global,'variance_rank_mean':variance_mean},Path(outdir)/f'{rank}.pt')
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('masks',[
    [[True,True],[True,False]],
    [[True,True],[False,False]],
    [[False,False],[False,False]],
])
def test_two_rank_masked_positive_gradient_equals_global_valid_mean(tmp_path,masks):
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from pathlib import Path
    from tools.streaming.train_ddp import masked_positive_cosine_loss
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip('Gloo unavailable')
    rendezvous=tmp_path/'gloo-store'
    mp.spawn(_distributed_positive_worker,args=(rendezvous,tmp_path,masks),nprocs=2,join=True)
    torch.manual_seed(412)
    layer=torch.nn.Linear(3,4,bias=False)
    inputs=torch.tensor([[.1,.2,.3],[.4,-.5,.6],[.3,.9,-.7],[.2,.1,-.8]])
    positives=torch.tensor([[.7,.2,.1,-.4],[.3,-.1,.4,.5],[.2,.7,-.3,.2],[-.5,.1,.9,.2]])
    loss=masked_positive_cosine_loss(layer(inputs),positives,torch.tensor(masks,dtype=torch.bool).flatten())
    loss.backward()
    for rank in (0,1):
        actual=torch.load(tmp_path/f'{rank}.pt',weights_only=True)
        torch.testing.assert_close(actual['gradient'],layer.weight.grad,rtol=1e-6,atol=1e-7)
        assert actual['positive_global']==pytest.approx(loss.item(),rel=1e-6,abs=1e-7)
        assert actual['variance_rank_mean']==pytest.approx(1.5)


def test_yaml_cli_resume_and_warm_start_override_each_other(tmp_path):
    config=tmp_path/'train.yml'
    config.write_text('synthetic: true\nrandom_visual: true\npositive_contrastive: true\noutput: out.pt\nwarm_start: first.pt\n')
    args=parse_args(['--config',str(config),'--resume','second.pt'])
    assert args.resume=='second.pt' and args.warm_start is None
    config.write_text('synthetic: true\nrandom_visual: true\npositive_contrastive: true\noutput: out.pt\nresume: first.pt\n')
    args=parse_args(['--config',str(config),'--warm-start','second.pt'])
    assert args.warm_start=='second.pt' and args.resume is None


def test_missing_training_metrics_remain_explicit():
    from tools.streaming.train_ddp import reduce_training_loss_metrics
    assert reduce_training_loss_metrics(None,None,torch.device("cpu")) == (None,None)
    assert reduce_training_loss_metrics(torch.tensor(.3),None,torch.device("cpu"))[1] is None


@pytest.mark.parametrize("value", [0.1, 0.2])
def test_float32_constant_detection_is_independent_of_mean_roundoff(value):
    from tools.streaming.train_ddp import target_degeneracy_masks
    target=torch.full((2,32000),value,dtype=torch.float32)
    target[1,0]=value+1e-4
    degenerate,silent=target_degeneracy_masks(target,0.0)
    assert degenerate.tolist()==[True,False] and silent.tolist()==[False,False]
    output=torch.full_like(target,.3,requires_grad=True)
    mixture=torch.ones_like(target)
    loss,_,_,d,_=reconstruction_components(output,target,mixture,0.0)
    assert d.tolist()==[True,False]
    torch.autograd.grad(loss[0],output)[0]


def test_resume_dual_validation_state_rejects_corruption():
    from tools.streaming.train_ddp import validation_tracking_from_checkpoint
    assert validation_tracking_from_checkpoint({'best_validation':-1.005,'significant_best_validation':-1.0,'stale_validations':2}) == (-1.005,-1.0,2)
    assert validation_tracking_from_checkpoint({'best_validation':float('inf'),'significant_best_validation':float('inf'),'stale_validations':0}) == (float('inf'),float('inf'),0)
    assert validation_tracking_from_checkpoint({'best_validation':-2.0,'significant_best_validation':float('inf'),'stale_validations':0}) == (-2.0,float('inf'),0)
    for bad in (
        {'best_validation':-1.,'stale_validations':0},
        {'best_validation':float('nan'),'significant_best_validation':-1.,'stale_validations':0},
        {'best_validation':-1.,'significant_best_validation':-2.,'stale_validations':0},
        {'best_validation':-1.,'significant_best_validation':-1.,'stale_validations':-1},
        {'best_validation':-1.,'significant_best_validation':-1.,'stale_validations':True},
        {'best_validation':float('inf'),'significant_best_validation':float('inf'),'stale_validations':1},
    ):
        with pytest.raises(ValueError,match='validation tracking state'):
            validation_tracking_from_checkpoint(bad)
