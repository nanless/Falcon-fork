import json
from pathlib import Path
p=Path("/tmp/falcon_full_review_20261110/verification-inventory.json");d=json.loads(p.read_text());m={x["path"]:x for x in d["files"]};root=Path("/root/code/github_repos/Falcon-fork")
for sub,csvname in [("DNSMOS-realssa-real-mean-full3","per_segment.csv"),("DNSMOS-realssa-real-mean-full4-20261031","realssa_real_dnsmos_per_segment.csv")]:
 prefix="docs/verification/phase2/"+sub+"/";summary=json.loads((root/prefix/"summary.json").read_text());a=m[prefix+csvname]["csv_numeric_audit"]
 comp={}
 for k,s in summary["metrics"].items():
  x=a["metrics"].get(k)
  if x and x["mean"] is not None:comp[k]={"n_csv":x["n_finite"],"n_summary":s["n"],"mean_from_rounded_csv":x["mean"],"mean_summary":s["mean"],"abs_difference":abs(x["mean"]-s["mean"])}
 for name in (prefix+csvname,prefix+"summary.json"):
  m[name]["dnsmos_crosscheck"]={"method":"unweighted mean of rounded per-segment values against saved summary; no rerun","results":comp,"protocol_status":"invalid_full3_historical" if "full3" in sub else "full4_valid_with_documented_protocol_limits"}
 print(sub, "N",summary["segments_total"],"max_abs_mean_difference",max(v["abs_difference"] for v in comp.values()),"ovrl",comp["ovrl_mos"],"p808",comp["p808_mos"])
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n")
