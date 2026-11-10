import ast,hashlib,json,pathlib,re
repo=pathlib.Path("/root/code/github_repos/Falcon-fork");root=repo/"docs/verification"
files=sorted(list((root/"scripts").rglob("*.py"))+list((root/"stubs").rglob("*.py")))
complete={"docs/verification/scripts/verify_r3_studio_weights.py","docs/verification/scripts/data_pipe/verify_realweight_eval.py","docs/verification/scripts/verify_macs_04.py"}
partial={"docs/verification/scripts/verify_studio_backend.py"}
rows=[]
for p in files:
 rel=str(p.relative_to(repo)); s=p.read_text();t=ast.parse(s)
 doc=(ast.get_docstring(t) or "").strip().split("\n")[0]
 first=[]
 for n in ast.walk(t):
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="print":
   for a in n.args:
    if isinstance(a,ast.Constant) and isinstance(a.value,str) and 4<len(a.value)<180 and a.value.strip():first.append(a.value.strip().replace("\n"," "));break
  if len(first)>=3:break
 tags=[]
 for pat,label in [(r"(?i)(stubs|pil_stub|sys\.modules|sys\.path\.insert)","stub_or_path"),(r"(?i)(falcon_weights|falcon\.pt|frcnn.*pth|torch\.load)","checkpoint_or_load"),(r"(?i)(cuda|device\s*=)","device_mentioned"),(r"(?i)(write_text|write_bytes|torch\.save|json\.dump|makedirs|mkdir|subprocess|os\.system|rmtree|unlink)","possible_write_or_process"),(r"(?i)(torch\.rand|np\.random|random\.|manual_seed)","random_input")]:
  if re.search(pat,s):tags.append(label)
 paths=sorted(set(re.findall(r"/(?:tmp|root|private)/[A-Za-z0-9_./{}*:-]+",s)))
 review="full" if rel in complete or "/stubs/" in rel or "/pil_stub/" in rel else "partial" if rel in partial else "static_scan"
 rows.append({"path":rel,"line_count":len(s.splitlines()),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"purpose":doc or ("; ".join(first) if first else "no module docstring; manual review required"),"markers":tags,"literal_paths":paths,"review_status":review})
out=pathlib.Path("/tmp/falcon_verification_inventory_20261110.json");out.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+"\n")
with open("/tmp/falcon_verification_inventory_20261110.txt","w") as f:
 f.write("Static inventory; markers are candidate indicators, not proven execution effects. No script executed.\n")
 for r in rows:f.write("{path} | lines={line_count} | {review_status} | {purpose} | markers={markers} | paths={literal_paths}\n".format(**r))
print("files",len(rows),"scripts",sum("/scripts/" in x["path"] for x in rows),"stubs",sum("/stubs/" in x["path"] for x in rows),"review_status",{s:sum(x["review_status"]==s for x in rows) for s in ["full","partial","static_scan"]})
print(out, "/tmp/falcon_verification_inventory_20261110.txt")
