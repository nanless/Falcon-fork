"""Audit metadata references against three ZIP central directories, without extraction."""
import argparse, collections, hashlib, json, zipfile
from pathlib import Path
r=Path('/aispeech-cfs/voiceprint/data/multimodal/mld_vc/raw')
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
if out.exists(): raise FileExistsError(out)
metadata=(r/'metadata.jsonl').read_bytes(); records=[json.loads(line) for line in metadata.splitlines() if line.strip()]
result={'date':'2026-11-11','source_metadata':str(r/'metadata.jsonl'),'metadata_sha256':hashlib.sha256(metadata).hexdigest(),'rows':len(records),'distinct_ids':len({d['id'] for d in records}),'distinct_speakers':len({d['speaker'] for d in records}),'fields':sorted(set().union(*(d.keys() for d in records))),'counts':{},'archive_checks':{},'scope':'All metadata rows and ZIP central-directory names. No extraction, audio/video sync or speaker identity validation.'}
for field in ('env','platform','noise_level'):
 result['counts'][field]=dict(collections.Counter(str(d[field]) for d in records))
for field,filename,prefix in [('audio','audio.zip','audio/'),('video','video.zip','video/'),('landmarks','landmarks.zip','landmarks/')]:
 with zipfile.ZipFile(r/filename) as z:
  names={i.filename for i in z.infolist() if not i.is_dir() and not i.filename.startswith('__MACOSX/') and '/._' not in i.filename}
  expected=[prefix+d[field].split('/',1)[1] for d in records]
  result['archive_checks'][field]={'archive':str(r/filename),'archive_bytes':(r/filename).stat().st_size,'file_count':len(names),'extensions':dict(collections.Counter(Path(x).suffix for x in names)), 'references':len(expected),'unique_references':len(set(expected)),'missing_references':sum(x not in names for x in expected)}
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print('saved',out,'rows',len(records),'speakers',result['distinct_speakers'])
