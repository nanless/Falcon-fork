from pathlib import Path
import tempfile
output=Path(tempfile.mkdtemp(prefix='falcon-summary-audit-',dir='/root/tmp'))/'result.json'
print('output',output)
import csv,json,math,statistics
root=Path('/root/code/github_repos/Falcon-fork/docs/verification/phase2'); checks=[]
def read(p):
 with p.open(newline='') as f: rows=list(csv.DictReader(f))
 return [r for r in rows if next(iter(r.values())) not in ('avg','std')]
def finite(rows,key):
 ret=[]
 for row in rows:
  try:x=float(row[key])
  except (ValueError,KeyError,TypeError): continue
  if math.isfinite(x):ret.append(x)
 return ret
for p in root.glob('P2B-summary-b300*.json'):
 for item in json.loads(p.read_text()):
  rows=read(root/item['csv'])
  for key in ('si_snr','si_snr_i','sdr','sdr_i','pesq','stoi'):
   vals=finite(rows,key.replace('si_snr','si-snr')); got=statistics.mean(vals)
   checks.append({'source':p.name,'csv':item['csv'],'metric':key,'difference':got-item[key], 'n_equal':len(rows)==item['n']})
for name,csvname in [('DNSMOS-realssa-real-mean-full3/summary.json','DNSMOS-realssa-real-mean-full3/per_segment.csv'),('DNSMOS-realssa-real-mean-full4-20261031/summary.json','DNSMOS-realssa-real-mean-full4-20261031/realssa_real_dnsmos_per_segment.csv'),('P2E-smoke-realssa-real-summary.json','P2E-smoke-realssa-real-per-segment.csv')]:
 p=root/name;d=json.loads(p.read_text());rows=read(root/csvname)
 for k,v in d['metrics'].items():
  vals=finite(rows,k); checks.append({'source':name,'csv':csvname,'metric':k,'difference':statistics.mean(vals)-v['mean'],'n_equal':len(vals)==v['n']})
for p in root.glob('*.counts.json'):
 d=json.loads(p.read_text());csvpath=p.with_name(p.name.replace('.counts.json','.csv')); rows=read(csvpath)
 for k,n in d['counts'].items():checks.append({'source':p.name,'csv':csvpath.name,'metric':k,'difference':0,'n_equal':len(finite(rows,k))==n})
fail=[x for x in checks if abs(x['difference'])>0.00011 or not x['n_equal']]
print('checks',len(checks),'failed',len(fail));print(json.dumps(fail,indent=2));output.write_text(json.dumps({'checks':checks,'failed':fail},ensure_ascii=False,indent=2))
