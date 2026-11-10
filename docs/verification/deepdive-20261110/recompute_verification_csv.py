import csv,json,math,statistics
from pathlib import Path
repo=Path("/root/code/github_repos/Falcon-fork")
p=Path("/tmp/falcon_full_review_20261110/verification-inventory.json");d=json.loads(p.read_text());count=0
metrics=["sdr","sdr_i","si-snr","si-snr_i","pesq","stoi","si-snr_mix","sdr_mix"]
for x in d["files"]:
 if x["kind"]!="csv":continue
 path=repo/x["path"]
 with path.open(newline="",encoding="utf-8-sig") as f:reader=csv.DictReader(f);rows=list(reader);fields=reader.fieldnames or []
 audit={"method":"finite numeric columns only; no audio/model inference; no protocol validation"}
 if "snt_id" in fields:
  data=[r for r in rows if r.get("snt_id") not in ("avg","std")];avg=[r for r in rows if r.get("snt_id")=="avg"];std=[r for r in rows if r.get("snt_id")=="std"]
  audit.update(utterance_rows=len(data),avg_rows=len(avg),std_rows=len(std),metrics={})
  for k in metrics:
   if k not in fields:continue
   values=[];bad=0
   for row in data:
    try:z=float(row[k])
    except (ValueError,TypeError):bad+=1;continue
    if math.isfinite(z):values.append(z)
    else:bad+=1
   if not values:continue
   calc=sum(values)/len(values);sigma=statistics.pstdev(values)
   def parsed(row,col):
    try:
     a=float(row.get(col,""));return a if math.isfinite(a) else None
    except (TypeError,ValueError):return None
   savedavg=parsed(avg[0],k) if len(avg)==1 else None;savedstd=parsed(std[0],k) if len(std)==1 else None
   rec={"n_finite":len(values),"n_nonfinite_or_missing":bad,"mean":calc,"std_population":sigma,"saved_avg":savedavg,"saved_std":savedstd,"abs_mean_error":abs(calc-savedavg) if savedavg is not None else None,"abs_std_error":abs(sigma-savedstd) if savedstd is not None else None}
   if "n_"+k in fields and len(avg)==1:rec["reported_n"] = parsed(avg[0],"n_"+k)
   audit["metrics"][k]=rec
  audit["status"]="means_and_population_std_recomputed";count+=1
 elif "ovrl_mos" in fields:
  audit["status"]="finite_mos_means_recomputed";audit["rows"]=len(rows);audit["metrics"]={}
  for k in ["sig_mos","bak_mos","ovrl_mos","p808_mos","mix_sig_mos","mix_bak_mos","mix_ovrl_mos","mix_p808_mos"]:
   if k not in fields:continue
   a=[]
   for row in rows:
    try:z=float(row[k]);a.append(z) if math.isfinite(z) else None
    except (ValueError,TypeError):pass
   audit["metrics"][k]={"n_finite":len(a),"n_other":len(rows)-len(a),"mean":sum(a)/len(a) if a else None}
  count+=1
 elif "train_loss_step" in fields:
  data=[]
  for r in rows:
   try:
    z=float(r["train_loss_step"]);st=float(r["step"])
    if math.isfinite(z) and math.isfinite(st):data.append((st,z))
   except (ValueError,TypeError):pass
  audit.update(status="training_step_columns_recomputed_only",rows=len(rows),finite_steps=len(data),step_first=data[0][0] if data else None,step_last=data[-1][0] if data else None,strictly_increasing=all(b[0]>a[0] for a,b in zip(data,data[1:])))
  count+=1
 else:audit["status"]="not_recognized_no_value_recomputation"
 x["csv_numeric_audit"]=audit
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n")
print("csv inspected",count)
for x in d["files"]:
 if x["kind"]!="csv":continue
 a=x["csv_numeric_audit"];mm=a.get("metrics",{}); bad=[(k,v.get("n_nonfinite_or_missing")) for k,v in mm.items() if v.get("n_nonfinite_or_missing")]; errors=[(k,max(v.get("abs_mean_error") or 0,v.get("abs_std_error") or 0)) for k,v in mm.items() if max(v.get("abs_mean_error") or 0,v.get("abs_std_error") or 0)>1e-5]
 print(x["path"],a["status"],"rows",a.get("utterance_rows",a.get("rows")),"nonfinite",bad,"diff>1e-5",errors)
