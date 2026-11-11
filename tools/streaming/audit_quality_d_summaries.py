"""Independently check D/A final training summaries and three eval summaries."""
from __future__ import annotations
import argparse,json,math,re
from pathlib import Path
from tools.streaming.quality_d_control_v2 import verify_d,compare,sha256
from tools.streaming.common import read_checkpoint


def audit_eval(e):
    rows=e['rows']; good=[x for x in rows if x['status']=='ok']; s=e['summary']
    if len(rows)!=512 or len(good)!=s.get('valid_pairs') or s.get('attempted')!=len(rows) or s.get('failed')!=len(rows)-len(good) or not good:
        raise ValueError('evaluation counts do not match rows')
    values={key:sum(float(r[key]) for r in good)/len(good) for key in ('falcon_si_sdri_db','streaming_si_sdri_db')}
    for key,value in values.items():
        actual=s.get(key+'_mean')
        if type(actual) not in (int,float) or not math.isfinite(actual) or not math.isclose(actual,value,rel_tol=1e-10,abs_tol=1e-10):
            raise ValueError('evaluation mean differs from rows: '+key)
    delta=s.get('streaming_minus_falcon_db')
    if type(delta) not in (int,float) or not math.isfinite(delta) or not math.isclose(delta,values['streaming_si_sdri_db']-values['falcon_si_sdri_db'],rel_tol=1e-10,abs_tol=1e-10):
        raise ValueError('evaluation difference differs from rows')
    return values


def audit_train(checkpoint, log_text, exit_text):
    if exit_text.strip()!='0':raise ValueError('training exit status is not zero')
    for k,v in {'steps':50,'end_reason':'max_steps','validation_samples':512,'validation_total':512,
                'validation_limit':0,'world_size':2,'batch_per_rank':1024,'gradient_accumulation':1}.items():
        if checkpoint.get(k)!=v:raise ValueError('training checkpoint summary mismatch: '+k)
    if checkpoint.get('resume_restarts_epoch') is not True or checkpoint.get('next_epoch')!=1:
        raise ValueError('expected 39+11 updates and partial second epoch')
    for k in ('target_learning_rate','learning_rate'):
        if not math.isclose(checkpoint.get(k,float('nan')),6.25e-5,rel_tol=1e-10):
            raise ValueError('training learning rate mismatch: '+k)
    lines=log_text.splitlines(); steps=[]; validation=[]
    for line in lines:
        step=re.match(r'epoch=(\d+) step=(\d+) ',line)
        if step: steps.append((int(step.group(1)),int(step.group(2))))
        match=re.match(r'epoch=(\d+) validation_samples=(\d+) validation_total=(\d+) validation_limit=(\d+) loss=([^ ]+) lr=([^ ]+) partial=(True|False)',line)
        if match:validation.append(match)
    if steps!=[(0,x) for x in range(1,40)]+[(1,x) for x in range(40,51)]:
        raise ValueError('training log has missing or unexpected steps')
    if len(validation)!=2:raise ValueError('expected complete and partial epoch validation')
    last=validation[-1]
    if last.group(1,2,3,4,7)!=('1','512','512','0','True'):
        raise ValueError('final training validation summary mismatch')
    if not math.isclose(float(last.group(5)),checkpoint['validation_loss'],abs_tol=5.1e-7):
        raise ValueError('training validation loss differs from checkpoint')
    if not math.isclose(float(last.group(6)),checkpoint['learning_rate'],rel_tol=1e-10):
        raise ValueError('training log learning rate differs from checkpoint')
    if not any(line.startswith('saved ') and Path(line[6:]).name=='final.pt' for line in lines):
        raise ValueError('training log lacks final save')
    return {'steps':len(steps),'validation_loss':checkpoint['validation_loss'],'validation_samples':512}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True)
    for k in ('initial','a','d'):p.add_argument('--'+k,required=True)
    p.add_argument('--comparison',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    root=Path(args.output_dir).resolve();manifest,abc,_=verify_d(root)
    eval_paths={n:Path(getattr(args,n)).resolve() for n in ('initial','a','d')}
    paths={**{'eval_'+n:p for n,p in eval_paths.items()},'comparison':Path(args.comparison).resolve(),
           'a_checkpoint':Path(abc['variants']['a_causal_positive']['config']).parent/'a_causal_positive/final.pt',
           'd_checkpoint':root/'final.pt','a_log':Path(manifest['abc_dir'])/'a_causal_positive/train.log',
           'd_log':root/'train.log','a_exit':Path(manifest['abc_dir'])/'a_causal_positive/exit.txt','d_exit':root/'exit.txt',
           'tool':Path(__file__).resolve()}
    before={k:sha256(v) for k,v in paths.items()}
    saved=json.loads(paths['comparison'].read_text())
    # Recompute all provenance, per-row scores and bootstrap with the saved seed.
    fresh=compare(root,eval_paths,seed=saved['bootstrap']['seed'],bootstrap=saved['bootstrap']['replicates'])
    if fresh!=saved:raise ValueError('saved comparison differs from independent recomputation')
    evaluations={n:audit_eval(json.loads(path.read_text())) for n,path in eval_paths.items()}
    training={n:audit_train(read_checkpoint(paths[n+'_checkpoint'],'cpu'),paths[n+'_log'].read_text(),paths[n+'_exit'].read_text()) for n in ('a','d')}
    verify_d(root)
    if any(sha256(v)!=before[k] for k,v in paths.items()):raise RuntimeError('audit input changed during validation')
    result={'format':'falcon-quality-d-summary-audit-v1','verified':True,
            'input_sha256':before,'training':training,'evaluation_means_from_rows':evaluations,
            'comparison_counts':fresh['counts'],'pairs':fresh['pairs']}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
