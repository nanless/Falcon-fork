import torch
import importlib.metadata
from ptflops import get_model_complexity_info
class Probe(torch.nn.Module):
 def __init__(self, mode): super().__init__(); self.mode=mode
 def forward(self, x):
  if self.mode=="matmul": return torch.matmul(x,x)
  if self.mode=="mul_function": return torch.mul(x,x)
  if self.mode=="mul_operator": return x*x
  if self.mode=="add_function": return torch.add(x,x)
  if self.mode=="add_operator": return x+x
  if self.mode=="stft": return torch.stft(x,n_fft=16,hop_length=8,return_complex=True)
  if self.mode=="istft": return torch.istft(x,n_fft=16,hop_length=8,length=64)
print("ptflops",importlib.metadata.version("ptflops"),"torch",torch.__version__)
for mode in ["matmul","mul_function","mul_operator","add_function","add_operator","stft","istft"]:
 x=torch.randn(1,8,8) if mode not in ("stft","istft") else torch.randn(1,64)
 if mode=="istft": x=torch.stft(x,n_fft=16,hop_length=8,return_complex=True)
 with torch.no_grad():
  macs,params=get_model_complexity_info(Probe(mode),(x,),input_constructor=lambda z:{"x":z[0]},as_strings=False,print_per_layer_stat=False,verbose=False)
 print(mode, "count=",macs,"params=",params)
