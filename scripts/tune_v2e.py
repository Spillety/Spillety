#!/usr/bin/env python3
"""## Grid: lr/rounds/min_data + time-decay weights on v2-full matrix."""

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score
from train_decision_path_v2 import load_enriched, retrieval_frames, sensitivity_block

from spillety.features.build import build_feature_matrix
from spillety.models.gbdt import _params, _scale_pos_weight

SEED = 72

X_raw, X_ctx, ctx_names, y, steps = load_enriched("data/elliptic_raw")
Z, D_ill, D_lic, I_ill, anchors_ill, freqs = retrieval_frames(X_raw, y, steps)
sens = sensitivity_block(Z, I_ill, anchors_ill)
X_new, _ = build_feature_matrix(
    {"distances": np.column_stack([D_ill, D_lic]), "anchor_type_freqs": freqs,
     "sensitivity": sens, "context": X_ctx},
    names={"anchor_types": ["illicit", "licit"], "context": ctx_names})

tr = steps <= 30
sel = (steps > 30) & (steps <= 35)
cal_m = (steps > 35) & (steps <= 40)
Xtrb, ytrb = X_new[tr | sel], y[tr | sel]
t_trb = steps[tr | sel]

results = []
for lr in (0.03, 0.05, 0.1):
    for rounds in (300, 500, 800):
        for min_data in (20, 50):
            p = _params(31, _scale_pos_weight(ytrb), SEED)
            p.update(learning_rate=lr, min_data_in_leaf=min_data)
            b = lgb.train(p, lgb.Dataset(Xtrb, label=ytrb), num_boost_round=rounds)
            s = average_precision_score(y[cal_m], b.predict(X_new[cal_m]))
            results.append((s, f"lr={lr} rounds={rounds} min_data={min_data}"))
            print(f"{s:.4f} lr={lr} rounds={rounds} min_data={min_data}", flush=True)

for lam in (0.05, 0.15):
    w = np.exp(-lam * (35 - t_trb))
    p = _params(31, _scale_pos_weight(ytrb), SEED)
    b = lgb.train(p, lgb.Dataset(Xtrb, label=ytrb, weight=w), num_boost_round=500)
    s = average_precision_score(y[cal_m], b.predict(X_new[cal_m]))
    results.append((s, f"decay lam={lam}"))
    print(f"{s:.4f} decay lam={lam}", flush=True)

results.sort(reverse=True)
print("BEST:", results[0])
