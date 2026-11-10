import ast, pathlib, numpy as np, json
p=pathlib.Path("/root/code/github_repos/Falcon-fork/tools/repro/phase2/protocol_check.py")
t=ast.parse(p.read_text())
nodes=[x for x in t.body if isinstance(x,ast.FunctionDef) and x.name in ("my_si_sdr","_best_lag")]
ns={"np":np}; exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),"exec"),ns)
rng=np.random.default_rng(20261110); ref=rng.standard_normal(4096); ref[:64]=0; ref[-64:]=0
rows=[]
for delay in (23,-23,0):
    est=np.zeros_like(ref)
    if delay>0: est[delay:]=ref[:-delay]
    elif delay<0: est[:delay]=ref[-delay:]
    else: est[:]=ref
    lag,zero,reported=ns["_best_lag"](ref,est)
    corrected=np.zeros_like(est)
    if lag>0: corrected[:-lag]=est[lag:]
    elif lag<0: corrected[-lag:]=est[:lag]
    else: corrected[:]=est
    rows.append(dict(constructed_est_delay_samples=delay,returned_lag=lag,zero_db=zero,current_alignment_db=reported,corrected_alignment_db=ns["my_si_sdr"](ref,corrected)))
print(json.dumps(rows,indent=2))
