from pathlib import Path
import json,csv,math,statistics,re,collections
root=Path('/root/code/github_repos/Falcon-fork/docs/verification'); mismatch=[]; report=[]; errors=[]
for p in sorted(root.rglob('*')):
 if not p.is_file():continue
 if p.suffix=='.json':
  try: json.loads(p.read_text(),parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
  except ValueError as e: errors.append((str(p.relative_to(root)),str(e)))
 if p.suffix!='.csv':continue
 with p.open(newline='') as f:rows=list(csv.DictReader(f))
 if not rows:continue
 field=list(rows[0])[0]; samples=[r for r in rows if r[field] not in ('avg','std')]; avgs=[r for r in rows if r[field]=='avg']; stds=[r for r in rows if r[field]=='std']; metrics={}
 for k in rows[0]:
  values=[]
  for r in samples:
   try:v=float(r[k])
   except (ValueError,TypeError):continue
   if math.isfinite(v):values.append(v)
  if values and re.search(r'(?i)(si.?snr|sdr|pesq|p808|ovrl|sig_mos|bak_mos)',k) and not k.startswith('n_'):
   m=statistics.mean(values);metrics[k]={'n':len(values),'mean':m,'excluded':len(samples)-len(values)}
   if avgs and avgs[0].get(k):
    try:
     stated=float(avgs[0][k]);delta=m-stated
     if math.isfinite(stated) and abs(delta)>0.00011:mismatch.append((str(p.relative_to(root)),k,'avg',m,stated,delta))
    except ValueError:pass
 report.append({'path':str(p.relative_to(root)),'sample_rows':len(samples),'summary_rows':len(rows)-len(samples),'metrics':metrics,'duplicate_id':len(samples)-len(set(r[field] for r in samples)) if field in ('snt_id','segment_id') else None})
print('strict_invalid_json',errors);print('avg_mismatch',mismatch);print('duplicates',[(x['path'],x['duplicate_id']) for x in report if x['duplicate_id']]);
for x in report:
 if x['path'].startswith('phase2/P2D-eval-') or x['path'].startswith('phase2/P2E-realssa-sim-metrics') or x['path'].startswith('phase2/P2E-realssa-sim-ch0-metrics'):
  print(x['path'],x['sample_rows'],x['summary_rows'], {k:v for k,v in x['metrics'].items() if k in ('si-snr_i','sdr_i','pesq')})
import tempfile
p=Path(tempfile.mkdtemp(prefix='falcon-csv-audit-',dir='/root/tmp'))/'result.json';p.write_text(json.dumps({'csv':report,'avg_mismatch':mismatch,'strict_invalid_json':errors},ensure_ascii=False,indent=2));print('out',p)
