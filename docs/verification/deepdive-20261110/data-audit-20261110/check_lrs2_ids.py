from pathlib import Path
import json,csv,hashlib
root=Path('/root/code/github_repos/Falcon-fork');index=Path('/root/data/falcon_lrs2_2mix/tt/mix.json'); ref=[Path(r[0]).name for r in json.loads(index.read_text())]; files=['P2D-eval-falcon-full.csv','P2D-eval-falcon-full-cols-20261031.csv','P2D-eval-epoch86.csv','P2D-eval-epoch99.csv','P2D-eval-epoch99-cols-20261031.csv','P2D-eval-epoch124-contrastive.csv'];out=[]
for f in files:
 p=root/'docs/verification/phase2'/f
 with p.open(newline='') as h:items=[r['snt_id'] for r in csv.DictReader(h) if r['snt_id'] not in ('avg','std')]
 out.append({'path':str(p.relative_to(root)),'csv_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'n':len(items),'duplicates':len(items)-len(set(items)),'missing':len(set(ref)-set(items)),'extra':len(set(items)-set(ref))})
x={'index':str(index),'index_sha256':hashlib.sha256(index.read_bytes()).hexdigest(),'index_n':len(ref),'index_duplicates':len(ref)-len(set(ref)),'results':out};print(json.dumps(x,ensure_ascii=False,indent=2));import tempfile
dest=Path(tempfile.mkdtemp(prefix='falcon-id-audit-',dir='/root/tmp'))/'result.json';print('output',dest)
with dest.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2)
