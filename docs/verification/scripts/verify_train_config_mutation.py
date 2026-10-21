import yaml, copy
import look2hear.models as M

cfg = yaml.safe_load(open('configs/Falcon.yml'))
a = cfg['audionet']['audionet_config']
before = {k: sorted(v.keys()) for k, v in a.items()
          if isinstance(v, dict) and k in ('audio_params', 'video_params', 'enc_dec_params')}
model = M.FalconSpeakerContrastive(sample_rate=16000, **a)
after = {k: sorted(v.keys()) for k, v in a.items()
         if isinstance(v, dict) and k in ('audio_params', 'video_params', 'enc_dec_params')}
print("audio_params before:", before['audio_params'])
print("audio_params after :", after['audio_params'])
print("video_params before:", before['video_params'])
print("video_params after :", after['video_params'])
print("enc_dec_params before:", before['enc_dec_params'])
print("enc_dec_params after :", after['enc_dec_params'])
print("total params:", sum(p.numel() for p in model.parameters()))
print("trainable:", sum(p.numel() for p in model.parameters() if p.requires_grad))
