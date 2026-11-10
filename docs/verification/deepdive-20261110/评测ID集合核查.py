import json
from pathlib import Path
import pandas as pd
pairs = [
    ('/root/data/falcon_lrs2_2mix/tt/mix.json', 'docs/verification/phase2/P2D-eval-falcon-full.csv'),
    ('/root/data/falcon_lrs2_2mix/tt/mix.json', 'docs/verification/phase2/P2D-eval-epoch124-contrastive.csv'),
    ('/root/data/falcon_realssa_sim/tt/mix.json', 'docs/verification/phase2/P2E-realssa-sim-metrics_combined.csv'),
    ('/root/data/falcon_realssa_sim_ch0/tt/mix.json', 'docs/verification/phase2/P2E-realssa-sim-ch0-metrics_combined-20261030.csv'),
]
for idx, path in pairs:
    planned = [Path(row[0]).name for row in json.load(open(idx))]
    df = pd.read_csv(path)
    actual = df.loc[~df.snt_id.astype(str).isin(['avg', 'std']), 'snt_id'].astype(str).tolist()
    a, b = set(planned), set(actual)
    print(path, 'index', len(planned), 'csv', len(actual), 'index_duplicates', len(planned)-len(a), 'csv_duplicates', len(actual)-len(b), 'missing', len(a-b), 'extra', len(b-a))
