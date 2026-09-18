# elliptic_v2c — компактные 7 фичей (2026-09-18)

Абляция: только `d_ill_min`, `d_lic_min`, ratio, frac, sensitivity + контекст. Test PR-AUC 0.641 — хуже полного v2 (0.649): сырые дистанции несут сигнал, резать нельзя. Скрипт: `scripts/train_decision_path_v2.py --feature-set compact`.
