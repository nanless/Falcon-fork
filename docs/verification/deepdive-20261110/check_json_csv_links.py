import json,math
from pathlib import Path
p=Path("/tmp/falcon_full_review_20261110/verification-inventory.json");d=json.loads(p.read_text());m={x["path"]:x for x in d["files"]};root=Path("/root/code/github_repos/Falcon-fork");phase="docs/verification/phase2/"
for jp in sorted((root/phase).glob("P2B-summary*.json")):
 rel=str(jp.relative_to(root));links=[]
 for row in json.loads(jp.read_text()):
  name=phase+row["csv"];a=m.get(name,{}).get("csv_numeric_audit",{});metric=a.get("metrics",{});delta={}
  for col,key in [("si-snr","si_snr"),("si-snr_i","si_snr_i"),("sdr","sdr"),("sdr_i","sdr_i"),("pesq","pesq"),("stoi","stoi")]:
   if col in metric and key in row:delta[key]=abs(metric[col]["mean"]-row[key])
  for col,key in [("si-snr_i","si_snr_i_std"),("sdr_i","sdr_i_std")]:
   if col in metric and key in row:delta[key]=abs(metric[col]["std_population"]-row[key])
  links.append({"csv":name,"rows_expected":row.get("n"),"rows_actual":a.get("utterance_rows"),"abs_errors":delta})
 m[rel]["crosscheck"]={"status":"summary_values_recomputed_from_referenced_csv","entries":links,"limitation":"no model inference or protocol verification"}
 print(rel,"entries",len(links),"max_error",max((max(q["abs_errors"].values()) for q in links if q["abs_errors"]),default=None))
for jp in sorted((root/phase).glob("P2D-eval-*.counts.json")):
 rel=str(jp.relative_to(root));obj=json.loads(jp.read_text());csv_rel=phase+jp.name.replace(".counts.json",".csv");a=m.get(csv_rel,{}).get("csv_numeric_audit",{});print(rel,"rows",obj.get("n_utterances"),"csv",a.get("utterance_rows"),"counts",obj.get("counts"));m[rel]["crosscheck"]={"status":"counts_compared_to_csv","counts_json":obj.get("counts"),"csv_rows":a.get("utterance_rows"),"csv_finite_counts":{k:v["n_finite"] for k,v in a.get("metrics",{}).items()},"limitation":"count and saved metric values only; no inference"}
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n")
