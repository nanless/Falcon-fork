"""Read only the first ten tar members of selected Mandarin AVSR bundles."""
import argparse,itertools,json,tarfile
from pathlib import Path
root=Path('/aispeech-cfs/voiceprint/data/multimodal/mandarin_avsr/raw')
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
if out.exists():raise FileExistsError(out)
rows=[]
for name in ('cv.tar.gz','tt.tar.gz','tr_1mix.tar.gz'):
 path=root/name
 with tarfile.open(path,'r|gz') as t:
  entries=[{'name':m.name,'bytes':m.size,'kind':'dir' if m.isdir() else 'file' if m.isfile() else 'other'} for m in itertools.islice(t,10)]
 rows.append({'archive':str(path),'archive_bytes':path.stat().st_size,'first_ten_members':entries})
result={'date':'2026-11-11','scope':'First ten sequential members of each compressed tar; not a count of complete splits.','archives':rows}
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print('saved',out)
