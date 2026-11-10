import csv,hashlib,json,subprocess,sys
from pathlib import Path
from collections import Counter
repo=Path("/root/code/github_repos/Falcon-fork")
tracked=subprocess.check_output(["git","ls-files","-z","docs/verification"],cwd=repo).decode().split("\0")
paths=[repo/x for x in tracked if x and Path(x).suffix.lower() in (".log",".csv",".json")]
rows=[]
for p in paths:
 rel=str(p.relative_to(repo));raw=p.read_bytes();suffix=p.suffix.lower();entry={"path":rel,"kind":suffix[1:],"family":str(p.relative_to(repo/"docs/verification").parent),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"review":"automatic_index_only","provenance_limit":"git_last_change does not establish execution code, data or environment"}
 git=subprocess.check_output(["git","log","-1","--format=%h %as %s","--",rel],cwd=repo,text=True).strip();entry["git_last_change"]=git
 try:
  if suffix==".json":
   obj=json.loads(raw);entry["parse"]="ok";entry["top_type"]=type(obj).__name__;entry["top_size"]=len(obj) if isinstance(obj,(list,dict)) else None
  elif suffix==".csv":
   with p.open(newline="",encoding="utf-8-sig") as f:
    rdr=csv.reader(f);head=next(rdr,None);n=0;mismatch=0
    for row in rdr:n+=1;mismatch+=len(row)!=len(head)
   entry.update(parse="ok",header_columns=len(head or []),data_rows=n,width_mismatch_rows=mismatch)
  else:
   t=raw.decode("utf8",errors="replace");entry.update(parse="text",lines=len(t.splitlines()),tracebacks=t.count("Traceback (most recent call last)"),complete="not_assessed")
 except Exception as e:entry.update(parse="error",parse_error=f"{type(e).__name__}: {e}")
 rows.append(entry)
summary={"tracked_count":len(rows),"counts":dict(Counter(x["kind"] for x in rows)),"parse_errors":[x["path"] for x in rows if x["parse"]=="error"],"csv_width_mismatches":[{"path":x["path"],"rows":x.get("width_mismatch_rows")} for x in rows if x.get("width_mismatch_rows")],"traceback_logs":[x["path"] for x in rows if x.get("tracebacks",0)],"method":"automatic metadata and syntax inventory; not semantic review, completeness/accuracy not established"}
out=Path("/tmp/falcon_full_review_20261110/verification-inventory.json");out.write_text(json.dumps({"summary":summary,"files":rows},ensure_ascii=False,indent=2)+"\n");print(json.dumps(summary,ensure_ascii=False,indent=2));print("out",out)
