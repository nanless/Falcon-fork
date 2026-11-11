import torch
from look2hear.models.FalconPaperStreamPrototype import PaperStreamPrototype

def test_output_and_speculative_rnn_state():
    torch.manual_seed(42)
    m=PaperStreamPrototype(width=8,rnn_hidden=8,heads=2).eval()
    a=torch.randn(1,3200)
    b=a.clone(); b[:,1920:]=torch.randn_like(b[:,1920:])*20
    video=torch.randn(1,3,512)
    with torch.no_grad():
        y,s=m(a,video); z,q=m(b,video)
    assert y.shape==(1,1920) and torch.isfinite(y).all()
    assert torch.allclose(s.temporal_h,q.temporal_h,atol=1e-6)
    assert torch.allclose(s.temporal_c,q.temporal_c,atol=1e-6)
    assert torch.allclose(s.memory,q.memory,atol=1e-6)
    assert torch.allclose(s.temporal_conv,q.temporal_conv,atol=1e-6)
    assert not torch.allclose(y,z)
    with torch.no_grad():
        out,s2=m(a,video,s)
    assert torch.isfinite(out).all() and s2.memory.shape[1]==30

def test_identity_mask_reconstructs_multiple_chunks():
    torch.manual_seed(3)
    m=PaperStreamPrototype(width=8,rnn_hidden=8,heads=2).eval()
    with torch.no_grad():
        m.mask.weight.zero_(); m.mask.bias.zero_()
        audio=torch.randn(1,1920*2+1280); video=torch.randn(1,3,512)
        y,s=m(audio[:,:3200],video)
        z,_=m(audio[:,1920:1920+3200],video,s)
    assert torch.allclose(y,audio[:,:1920],atol=2e-5)
    assert torch.allclose(z,audio[:,1920:3840],atol=2e-5)

def test_gradient_across_chunks_and_future_limit():
    torch.manual_seed(5)
    m=PaperStreamPrototype(width=8,rnn_hidden=8,heads=2).train()
    audio=torch.randn(1,1920*2+1280,requires_grad=True)
    video=torch.randn(1,3,512,requires_grad=True)
    first,s=m(audio[:,:3200],video)
    g=torch.autograd.grad(first.square().mean(),audio,retain_graph=True)[0]
    assert g[:,3200:].count_nonzero()==0
    assert g[:,1920:3200].abs().sum()>0
    second,_=m(audio[:,1920:5120],video,s)
    (first.square().mean()+second.square().mean()).backward()
    assert torch.isfinite(audio.grad).all() and audio.grad.abs().sum()>0
    for name in ('input_conv.weight','freq_rnn.weight_ih_l0','time_rnn.weight_ih_l0','mask.weight'):
        grad=dict(m.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0


def test_dilated_conv_state_matches_continuous_reference():
    torch.manual_seed(19)
    m=PaperStreamPrototype(width=8,rnn_hidden=8,heads=2,temporal_kernel=5,temporal_dilation=16)
    width=8; n=3; frames=15; cache=torch.zeros(1,width,m.temporal_context)
    seq=torch.randn(1,width,n*frames)
    expected=m.temporal_depthwise(torch.cat((cache,seq),-1))
    got=[]
    for i in range(n):
        x=seq[:,:,i*frames:(i+1)*frames]
        joined=torch.cat((cache,x),-1)
        got.append(m.temporal_depthwise(joined))
        cache=joined[:,:,-m.temporal_context:]
    assert torch.allclose(torch.cat(got,-1), expected, atol=1e-6)
    assert cache.shape[-1] == 64

def test_checkpoint_preserves_state_gradients():
    import copy
    from torch.utils.checkpoint import checkpoint
    torch.manual_seed(29)
    base=PaperStreamPrototype(width=8,rnn_hidden=8,heads=2)
    tested=copy.deepcopy(base)
    a=torch.randn(1,1920*2+1280)
    v=torch.randn(1,3,512)
    outputs=[]
    for model,recompute in [(base,False),(tested,True)]:
        state=None; pieces=[]
        for i in range(2):
            inputs=(a[:,i*1920:i*1920+3200],v,state)
            out,state=checkpoint(model,*inputs,use_reentrant=False) if recompute else model(*inputs)
            pieces.append(out)
        result=torch.cat(pieces,dim=-1)
        result.square().mean().backward()
        outputs.append(result)
    assert torch.allclose(*outputs,atol=1e-6,rtol=1e-5)
    for (name,p),(other,q) in zip(base.named_parameters(),tested.named_parameters()):
        assert name==other
        assert (p.grad is None)==(q.grad is None), name
        if p.grad is not None:
            assert torch.allclose(p.grad,q.grad,atol=1e-6,rtol=1e-4),name

def test_cuda_bfloat16_two_chunks_and_checkpoint():
    import pytest
    from torch.utils.checkpoint import checkpoint
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip('CUDA BF16 unavailable')
    torch.manual_seed(42)
    m=PaperStreamPrototype(width=8,rnn_hidden=8,heads=2).cuda().train()
    audio=torch.randn(2,5120,device='cuda')
    video=torch.randn(2,3,512,device='cuda')
    state=None; out=[]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for i in range(2):
            y,state=checkpoint(m,audio[:,i*1920:i*1920+3200],video,state,use_reentrant=False)
            assert y.dtype==torch.float32
            assert state.temporal_h.dtype==torch.bfloat16
            assert state.temporal_c.dtype==torch.bfloat16
            assert state.temporal_conv.dtype==torch.bfloat16
            assert state.memory.dtype==torch.bfloat16
            out.append(y)
        loss=torch.cat(out,dim=1).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    for name,param in m.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(),name
