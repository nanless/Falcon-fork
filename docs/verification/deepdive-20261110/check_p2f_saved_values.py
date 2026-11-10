import json
from pathlib import Path
root=Path("/root/code/github_repos/Falcon-fork"); rel="docs/verification/phase2/P2F-results.json"; m=json.loads((root/rel).read_text())["metrics"]
res={"n":len(m["per_item"]),"metrics":{}}
for k in ("si_sdr","si_sdr_i","sdr","sdr_i"):
 v=[float(x["my"][k]) for x in m["per_item"]];delta=[abs(float(x["my"][k])-float(x["core"][k])) for x in m["per_item"]]
 res["metrics"][k]={"mean_my":sum(v)/len(v),"maxabs_my_core":max(delta)}
p=Path("/tmp/falcon_full_review_20261110/verification-inventory.json");d=json.loads(p.read_text());x=next(x for x in d["files"] if x["path"]==rel)
x["crosscheck"]={"status":"saved_per_item_values_recomputed","details":res,"limitation":"arithmetic only, no audio/model inference; no validation of every ablation"}
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n");print(json.dumps(res,indent=2))
