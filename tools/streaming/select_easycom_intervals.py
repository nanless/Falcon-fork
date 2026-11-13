import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Find EasyCom overlap and visible-silence intervals")
    parser.add_argument("dataset_root", type=Path, help="EasyCom Main directory")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    p = args.dataset_root
    out=[]
    for path in sorted((p/'Speech_Transcriptions').glob('*/*.json')):
        rel=path.relative_to(p/'Speech_Transcriptions')
        face=p/'Face_Bounding_Boxes'/rel
        if not face.is_file(): continue
        try:
            segments=json.loads(path.read_text())
            faces=json.loads(face.read_text())
        except (OSError,ValueError):
            continue
        total=max(x['Frame_Number'] for x in faces)+1
        active={}
        for t in segments:
            a,b=int(t['Start_Frame']),int(t['End_Frame'])
            who=t['Participant_ID']
            tx=t['Transcription'].strip()
            if not tx or all(part.startswith('[') for part in tx.split()): continue
            v=active.setdefault(who,np.zeros(total,dtype=bool))
            v[max(1,a):min(total,b+1)]=True
        if len(active)<3: continue
        visible={}
        for frame in faces:
            for box in frame['Participants']:
                if min(box['x2']-box['x1'],box['y2']-box['y1'])<65: continue
                v=visible.setdefault(box['Participant_ID'],np.zeros(total,dtype=bool))
                v[frame['Frame_Number']]=True
        number=np.stack(list(active.values())).sum(0)
        for who,v in visible.items():
            if who not in active: continue
            speak=active[who]
            for k,mask in [('overlap3',v & speak & (number>=3)),('visible_silent',v & ~speak & (number>=1))]:
                transitions=np.diff(np.r_[False,mask,False].astype('int8'))
                spans=list(zip(np.flatnonzero(transitions==1),np.flatnonzero(transitions==-1)))
                for a,b in spans:
                    if b-a < (5 if k=='overlap3' else 18): continue
                    if k=='visible_silent' and np.any(speak[max(1,a-6):min(total,b+6)]): continue
                    out.append({'type':k,'session':rel.parent.name,'file':path.stem,'target':who,
                         'start_frame':int(a),'end_frame_excl':int(b),'n_frames':int(b-a),
                         'active_others':[int(i) for i,act in active.items() if i!=who and np.count_nonzero(act[a:b])],
                         'annotation':str(path),'face_boxes':str(face)})
    out.sort(key=lambda d:(d['type'],-d['n_frames'],d['session'],d['file']))
    with args.output.open('x') as stream:
        json.dump(out, stream, indent=2)
    print(Counter(x['type'] for x in out))
    for x in [y for y in out if y['type']=='overlap3'][:12]: print(x['session'],x['file'],x['target'],x['start_frame'],x['n_frames'],x['active_others'])


if __name__ == "__main__":
    main()
