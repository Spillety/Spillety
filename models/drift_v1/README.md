# Drift v1 — epoch scan, steps 31–49 (seed 72, base = steps ≤ 30)

Source: `scripts/run_drift.py` → `report.json`. PCA32 fit on base only, 2000 base
anchors; no future data in any reference (past-only base, §8/§9).

## What happened at step 43

Not a geometry break — a label/score collapse:

- Walk-forward PR-AUC: 0.897 (step 42) → **0.047** (step 43), stays ≤ 0.16 through
  step 49 (one bounce: 0.508 at step 46, n = 712).
- Base rate: 0.111 → **0.018** at step 43, down to 0.003 by step 46 — the
  illicit minority the scorer was calibrated on nearly vanishes.
- Embedding median distance at step 43 is 16.42 vs τd = 17.93 (Q0.95 of base
  epoch medians): high but **under** the gate; no epoch 31–49 crosses τd.
  Regime at 43 stays `normal/incremental` — geometry gates miss it.
- Pooled steps 43–49 vs base: 36/173 features drift (KS+Cohen joint rule),
  top `feat_102` (D = 1.000, d = −5.20), `feat_104` (0.999, −5.07),
  `feat_101` (0.999, −4.48) — near-complete separation on a few raw features.

## When retrain fires

- `walk_forward_retrain` (rel. drop > 10% vs step 31): True from step 37 on
  (0.874 → 12.6% drop), and for every step 43–49.
- `retrain_gate`: True globally — rel. drop 83.7% (1.000 → 0.163) plus
  Mann-Kendall S = −109, p = 1.5e−04 (significant downward trend).
- Regime actions alone (`noise/ignore`, one `novel/review` at step 48) never
  trigger retrain — the walk-forward/PR-AUC gate is the one that catches
  step 43. Retrain decision: **yes, at step 43**.
