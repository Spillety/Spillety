# 10.2 Model Validation — Design

> Phase: D (Design) | Slug: model-validation | Status: In Progress

## 1. Baselines: Heuristic, LR, GCN, GAT

Benchmark suite: heuristic rules, logistic regression, GCN, GAT. AUC-PR primary metric.

### [human] Baseline selection
> Какие baseline модели тестировать?

**Decision:** 4 baselines: (1) heuristic (heuristic rules from domain experts), (2) LR (logistic regression — linear baseline), (3) GCN (Graph Convolutional Network), (4) GAT (Graph Attention Network). # ponytail: additional baselines, add when GAT < LR AUC-PR.

### [agent] resolved
> Heuristic, LR, GCN, GAT. GAT expected to outperform GCN for attention-based edge weighting.

### [human] Heuristic rules specification
> Какие heuristic rules использовать?

**Decision:** Co-spending heuristic (Bitcoin-style), exchange withdrawal pattern, mixer interaction flag, rapid velocity threshold. # ponytail: rule count, add when heuristic AUC-PR < 0.5.

### [agent] resolved
> 4 heuristic rules. AUC-PR baseline ~0.55-0.65.

## 2. Metrics: AUC-PR, Calibration Error, Fairness

AUC-PR для imbalanced classes. Calibration error для reliable scores. Fairness для bias detection.

### [human] Why AUC-PR not AUC-ROC?
> AUC-ROC или AUC-PR для AML?

**Decision:** AUC-PR. AML — сильно сбалансированный класс (legitimate >> scam). AUC-ROC оптимистичен при imbalance. AUC-PR — честная метрика для положительного класса. # ponytail: metric choice, add when positive class > 10%.

### [agent] resolved
> AUC-PR как primary. AUC-ROC supplementary для reference.

### [human] Calibration error
> Зачем нужна calibration?

**Decision:** Calibration error гарантирует что score=0.9 означает реальную вероятность ~90%. Для regulatory reporting. Platt scaling или isotonic regression. # ponytail: calibration method, add when calibration error > 0.05.

### [agent] resolved
> Platt scaling для calibration. Error < 0.05 threshold.

## 3. Bootstrap Confidence Intervals & Human Eval A/B

Bootstrap CI для статистической значимости. Human A/B testing для analyst preference.

### [human] Bootstrap CI approach
> Как построить confidence intervals для ML метрик?

**Decision:** Bootstrap 1000 resamples. 95% CI для AUC-PR. Оценка стабильности модели на разных подвыборках. # ponytail: bootstrap iterations, add when CI width > 0.05.

### [agent] resolved
> 1000 bootstrap iterations. 95% CI. Width < 0.05.

### [human] Human A/B testing design
> Как организовать human evaluation?

**Decision:** A/B testing analysts: случайная порция alerts от model A vs B. Предпочтение, time-to-decision, false positive rate. Double-blind. # ponytail: sample size, add when analyst pool < 20.

### [agent] resolved
> Double-blind A/B. Минимум 20 analysts. Предпочтение + time-to-decision.

## 4. Pushback: What Could Go Wrong

### [human] Hypothesis 1: GAT overfits on small graph datasets
> GAT может переобучиться на ограниченном AML данных.

**Decision:** Dropout (p=0.5) + early stopping на validation AUC-PR. GCN как fallback если GAT не улучшает LR на > 2%. # ponytail: GAT vs GCN, add when GAT < GCN AUC-PR.

### [agent] resolved
> Dropout + early stopping. GCN fallback threshold 2%.

### [human] Hypothesis 2: Human A/B testing introduces bias
> Analyst preference может быть смещена.

**Decision:** Double-blind, randomized alert assignment. Analyst doesn't know which model generated the alert. Statistical significance test (Mann-Whitney U). # ponytail: bias control, add when preference gap < 5%.

### [agent] resolved
> Double-blind + Mann-Whitney U test. Preference gap < 5% = inconclusive.

## 5. Acceptance Criteria

- [ ] Baselines: heuristic, LR, GCN, GAT — all benchmarked
- [ ] AUC-PR as primary metric (AUC-ROC supplementary)
- [ ] Calibration error < 0.05 (Platt scaling)
- [ ] Fairness metrics computed across demographics
- [ ] Bootstrap CI: 1000 iterations, 95% CI
- [ ] Human A/B: double-blind, > 20 analysts
- [ ] Self-review: 0 🔴 Critical, 0 ai-slops, AC выполнены

## 6. Pushback Summary

| # | Гипотеза | Решение |
|---|----------|---------|
| 1 | GAT overfits on small data | Dropout + early stopping, GCN fallback |
| 2 | Human A/B introduces bias | Double-blind + Mann-Whitney U |
