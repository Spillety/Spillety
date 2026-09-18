# Causal v1 — DAG-select on real data (J2 wave J, §6.3)

Best: **DAG-E** (status ok), train n=31235 (steps ≤35 labeled), test n=9973 (steps 41–49).

## Rejected edges

### DAG-A (C=[]): 3 rejections
- `exchange_hot_||_y|[]`: falsified — absent var carries signal beyond C (fals p=1.55e-98).
- `mixer_||_y|[]`: falsified — absent var carries signal beyond C (fals p=7.25e-54).
- `hawkes_||_y|[]`: falsified — absent var carries signal beyond C (fals p=5.04e-58).

### DAG-B (C=['exchange_hot']): 4 rejections
- `feat_3->y|['exchange_hot']`: unsupported — ci p=0.719, CI=[0.00982,0.969] ≥ α (edge adds nothing beyond C).
- `feat_5->y|['exchange_hot']`: unsupported — ci p=0.0105, CI=[7.47e-06,0.498] ≥ α (edge adds nothing beyond C).
- `mixer_||_y|['exchange_hot']`: falsified — absent var carries signal beyond C (fals p=3.39e-06).
- `hawkes_||_y|['exchange_hot']`: falsified — absent var carries signal beyond C (fals p=1.33e-52).

### DAG-C (C=['exchange_hot', 'mixer']): 3 rejections
- `feat_3->y|['exchange_hot', 'mixer']`: unsupported — ci p=0.488, CI=[0.00404,0.966] ≥ α (edge adds nothing beyond C).
- `feat_5->y|['exchange_hot', 'mixer']`: unsupported — ci p=0.0415, CI=[7.22e-05,0.755] ≥ α (edge adds nothing beyond C).
- `hawkes_||_y|['exchange_hot', 'mixer']`: falsified — absent var carries signal beyond C (fals p=6.27e-52).

### DAG-D (C=['exchange_hot', 'mixer', 'hawkes']): 3 rejections
- `feat_3->y|['exchange_hot', 'mixer', 'hawkes']`: unsupported — ci p=0.963, CI=[0.0113,0.97] ≥ α (edge adds nothing beyond C).
- `feat_5->y|['exchange_hot', 'mixer', 'hawkes']`: unsupported — ci p=0.0269, CI=[3.32e-05,0.752] ≥ α (edge adds nothing beyond C).
- `time_step_||_y|['exchange_hot', 'mixer', 'hawkes']`: falsified — absent var carries signal beyond C (fals p=1.52e-148).

### DAG-E (C=['hawkes']): 2 rejections
- `exchange_hot_||_y|['hawkes']`: falsified — absent var carries signal beyond C (fals p=4e-93).
- `mixer_||_y|['hawkes']`: falsified — absent var carries signal beyond C (fals p=1.03e-49).

## Stability (causal_filter pass_rate, selected DAG)
- train: 1.000, test: 0.750
- E-value p50 train/test: 1.67 / 1.55; Tier1 (E>2) share train/test: 0.000 / 0.250

## Notes
- Selection stats use train era only (steps ≤35 labeled); no future rows in ci/falsification/select.
- Hub flags are label-free structural proxies (full-graph degree/PageRank, top 5%); Hawkes λ is causal-in-time (past counts only).
- Bootstrap n=1000.
