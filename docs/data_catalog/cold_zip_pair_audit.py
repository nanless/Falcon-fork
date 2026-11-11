"""Read ZIP central directories from the cold mount; compare audiovisual names."""
import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import zipfile

ROOT=Path('/aispeech-cfs/voiceprint/data/multimodal')
PAIR_CASES=[
 ('aspire','aspire/raw/ASPIRE.zip','.wav','aspire/raw/ASPIRE.zip','.mov','stem'),
 ('chinese_train','chinese_lips/raw/processed_train.zip','.wav','chinese_lips/raw/processed_train.zip','.mp4','stem'),
 ('chinese_val','chinese_lips/raw/processed_val.zip','.wav','chinese_lips/raw/processed_val.zip','.mp4','stem'),
 ('chinese_test','chinese_lips/raw/processed_test.zip','.wav','chinese_lips/raw/processed_test.zip','.mp4','stem'),
 ('lombard_front','lombard_grid/raw/lombardgrid_audio.zip','.wav','lombard_grid/raw/lombardgrid_front.zip','.mov','stem'),
 ('mld_vc','mld_vc/raw/audio.zip','.wav','mld_vc/raw/video.zip','.mp4','relative_after_top'),
]
EXTRA_ARCHIVES=['cuave_partial/raw/cuave-group-aligned.zip','grid/raw/s1.zip']

def is_media_candidate(name):
    return '__MACOSX/' not in name and '/._' not in name and not Path(name).name.startswith('._')

def read_zip(root, rel):
    path=root/rel
    with zipfile.ZipFile(path) as z:
        names=[i.filename for i in z.infolist() if not i.is_dir() and is_media_candidate(i.filename)]
    return {'path':str(path),'archive_bytes':path.stat().st_size,'entry_count':len(names),
            'extensions':dict(Counter(Path(n).suffix.lower() or '[no extension]' for n in names)),
            'examples':names[:3]}, names

def key(name, kind):
    p=Path(name)
    return p.stem if kind=='stem' else str(Path(*p.parts[1:]).with_suffix(''))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    now=datetime.now(timezone.utc).isoformat(); cache={}
    def get(rel):
        if rel not in cache:cache[rel]=read_zip(a.root,rel)
        return cache[rel]
    comparisons=[]
    for label,ap,ae,vp,ve,kind in PAIR_CASES:
        _,aa=get(ap);_,vv=get(vp)
        ca=Counter(key(n,kind) for n in aa if n.lower().endswith(ae));cv=Counter(key(n,kind) for n in vv if n.lower().endswith(ve))
        comparisons.append({'dataset':label,'audio_archive':ap,'video_archive':vp,'comparison_key':kind,
          'audio_files':sum(ca.values()),'video_files':sum(cv.values()),'audio_unique_keys':len(ca),'video_unique_keys':len(cv),
          'matched_unique_keys':len(ca.keys()&cv.keys()),'audio_only_keys':len(ca.keys()-cv.keys()),'video_only_keys':len(cv.keys()-ca.keys()),
          'audio_duplicate_keys':sum(v>1 for v in ca.values()),'video_duplicate_keys':sum(v>1 for v in cv.values())})
    for rel in EXTRA_ARCHIVES:get(rel)
    result={'captured_at_utc':now,'scope':'ZIP central directory only; filename matching does not verify waveform, timestamp synchronization, speaker identity or media quality; no extraction or media hashes',
            'archives':[v[0] for v in cache.values()],'comparisons':comparisons}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'archives':len(cache),'comparisons':comparisons},ensure_ascii=False))
if __name__=='__main__':main()
