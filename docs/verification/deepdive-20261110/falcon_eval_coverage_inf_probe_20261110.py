"""Synthetic-only evidence: run from repo with PYTHONDONTWRITEBYTECODE=1 via python -c exec."""
import csv, json, tempfile, pathlib, hashlib
import pandas as pd
import audio_visual_test as av
from look2hear.metrics.allwrapper import ALLMetricsTracker
root = pathlib.Path(tempfile.mkdtemp(prefix="falcon_coverage_inf_probe_"))
results = root / "results"; results.mkdir()
plan={"dataset_len":4,"limit":0,"world_size":2,"planned_total":4,"planned_per_rank":[2,2],"rank0_evaluated":2}
(results / "coverage_rank0.json").write_text(json.dumps(plan))
cols=["snt_id","si-snr","n_si-snr"]
for rank, rows in enumerate(([("A",1),("B",2)],[("A",3),("D",float("inf"))])):
 with (results / f"metrics_rank{rank}.csv").open("w",newline="") as f:
  w=csv.writer(f);w.writerow(cols);w.writerows([[k,v,""] for k,v in rows]);w.writerows([["avg",0,2],["std",0,2]])
av._merge_results(str(root),2,enforce_coverage=True)
out=pd.read_csv(results/"metrics_combined.csv")
report=json.loads((results/"metrics_counts.json").read_text())
ids=out.loc[~out.snt_id.isin(["avg","std"]),"snt_id"].tolist()
print("source_sha256",hashlib.sha256(pathlib.Path(av.__file__).read_bytes()).hexdigest())
print("expected_ids",["A","B","C","D"],"actual_ids",ids,"unique",len(set(ids)))
print("coverage",json.dumps(report["coverage"],sort_keys=True))
print("merged_count",report["counts"]["si-snr"],"merged_nan_count",report["nan_counts"]["si-snr"])
print("merged_avg",out.loc[out.snt_id=="avg","si-snr"].iloc[0])
vals=[1.0,2.0,3.0,float("inf")]
tracker=object.__new__(ALLMetricsTracker);tracker._value_attr=lambda: (("si-snr","values"),);tracker.values=vals
print("tracker_count",tracker.get_counts()["n_si-snr"],"tracker_mean",tracker.get_mean()["si-snr"])
print("temp_root",root)
assert report["coverage"]["verified"] is True and set(ids)!={"A","B","C","D"}
assert report["counts"]["si-snr"] == 4 and tracker.get_counts()["n_si-snr"] == 3
