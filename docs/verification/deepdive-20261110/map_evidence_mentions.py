import json
from pathlib import Path
from collections import Counter
repo=Path("/root/code/github_repos/Falcon-fork");p=Path("/tmp/falcon_full_review_20261110/verification-inventory.json");d=json.loads(p.read_text());names=Counter(Path(x["path"]).name for x in d["files"])
md=[]
for f in (repo/"docs").rglob("*.md"):
 try:md.append((str(f.relative_to(repo)),f.read_text(encoding="utf-8",errors="replace")))
 except OSError:pass
for x in d["files"]:
 path=Path(x["path"]);key=path.name if names[path.name]==1 else "/".join(path.parts[-2:]);hits=[]
 for file,text in md:
  if key in text:hits.append(file)
 x["markdown_mentions"]={"query":key,"documents":hits,"method":"literal string occurrence only; references are not checked or endorsed"}
d["summary"]["markdown_scan_files"]=len(md);d["summary"]["files_with_literal_markdown_mentions"]=sum(bool(x["markdown_mentions"]["documents"]) for x in d["files"])
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n")
print("docs scanned",len(md),"evidence with mentions",d["summary"]["files_with_literal_markdown_mentions"])
