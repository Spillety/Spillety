# elliptic_v2 — retrieval + causal-признаки, refit (2026-09-18)

Промежуточная точка: d=206 (20 дистанций + frac + sensitivity + 171 контекст), протокол B (select 31–35, refit train+select, калибровка 36–40). Test PR-AUC 0.649 (протокол A без refit — 0.56). Заменена финалом `models/elliptic_v3/`. Скрипт: `scripts/train_decision_path_v2.py`.
