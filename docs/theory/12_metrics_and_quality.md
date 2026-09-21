# 12. Метрики и оценка качества

У каждого компонента Spillety есть локальные метрики, но система в целом отвечает на другой вопрос: что она реально делает и чего не делает. Эта глава собирает единый метрический контур — что измеряем, на чём, против какого baseline и с какими честными оговорками. Принцип прост: любое заявление сопровождается base rate и доверительным интервалом, а всё, что не может быть измерено, не заявляется вовсе.

## 12.1. Требования к контуру

Метрический контур обязан: отражать фактические характеристики без завышенных заявлений; быть сравнимым с baseline (тривиальный предиктор, XGBoost на Elliptic++); оцениваться на hold-out с временной изоляцией; разделять retrieval, скоринг, калибровку, операционный и стоимостной контуры и self-evolution; явно учитывать base rate. Последний пункт не формальность: precision при base rate 10% и при 2% — несравнимые числа, и без нормализации сравнение бессмысленно.

## 12.2. Таксономия

| Группа | Объект измерения | Метрики |
|--------|------------------|---------|
| Качество ранжирования | Разделение illicit/legit | PR-AUC, Precision@K, Recall@K |
| Калибровка | Соответствие вероятностей частотам | Brier, ECE |
| Операционные | Поведение в проде | FP-rate на аналитика, alert-to-SAR, TTD, latency p99 |
| Стоимость | Экономика | labeling cost, FTE, cost per alert |
| Self-evolution | Адаптивность | median distance, drift KS, recall@K на новых санкциях |
| Доверие и соответствие | Проверяемость | audit verification, Merkle proof size, SR 26-2 validation, bias equalized odds |

## 12.3. Качество ранжирования

### 12.3.1. PR-AUC

$$
\text{PR-AUC} = \int_0^1 \text{Precision}(\text{Recall}) \, d\text{Recall}
$$

В отличие от ROC-AUC, PR-AUC чувствителен к base rate — и это именно то, что нужно при редком положительном классе: высокий ROC-AUC может маскировать катастрофически низкий precision. Оценка — hold-out temporal split (train на прошлом, test на будущем), кривая на test; baseline — XGBoost на Elliptic++ (56 признаков). Важная оговорка: PR-AUC показывает потенциал модели, но не заменяет precision при рабочем пороге — операционное качество читается в точке $\tau^*$ из раздела 8.

### 12.3.2. Precision@K и Recall@K

$$
\text{Precision@K} = \frac{|\{\text{relevant}\} \cap \{\text{top-}K\}|}{K}, \qquad
\text{Recall@K} = \frac{|\{\text{relevant}\} \cap \{\text{top-}K\}|}{|\{\text{relevant}\}|}
$$

Для retrieval: доля истинно связанных якорей среди top-$K$ и полнота покрытия истинных соседей. $K$ выбирается по cost-функции, отчёт сопровождается base rate. Recall@K оценивается на уровне кластеров (более стабильная единица, глава 5). Отдельный срез — temporal: recall@K на санкциях, добавленных после даты обучения, показывает обобщение на невиданные якоря. Доверительные интервалы — bootstrap CI.

## 12.4. Калибровка

### 12.4.1. Brier и ECE

$$
\text{Brier} = \frac{1}{n}\sum_{i=1}^{n}(\hat{p}_i - y_i)^2, \qquad
\text{ECE} = \sum_{m=1}^{M}\frac{|B_m|}{n}\left|\text{acc}(B_m) - \text{conf}(B_m)\right|
$$

Brier: 0 — идеально, 0.25 — случайный предиктор на сбалансированных классах; при дисбалансе метрика смещена к base rate, поэтому обязательно сравнение с тривиальным предиктором $\hat{p}_i \equiv \bar{y}$. ECE: $B_m$ — бины ($M=10$–$15$), значения выше $0.1$ — плохая калибровка.

### 12.4.2. Выбор калибратора

Isotonic против beta: непараметрическая гибкость против параметрической устойчивости на малых выборках. Решение принимается по ECE/Brier на hold-out temporal split — минимальный ECE при стабильном Brier, проверка bootstrap CI для разности, сохранение recall@K при фиксированном ECE; надёжность подтверждается reliability diagram. Полная процедура — в разделе 7.

## 12.5. Операционные метрики

**FP-rate на аналитика**: $\text{FP-rate} = \frac{\text{FP}}{\text{TP}+\text{FP}}$ на аналитика в единицу времени; источник — операционные данные с разметкой аналитика. Метрика связана с cost-функцией напрямую: рост FP-rate сдвигает оптимальный порог вправо.

**Alert-to-SAR conversion**: $\frac{\#\text{SAR}}{\#\text{alerts}}$ — полезность алертов для compliance; низкая конверсия — индикатор шума. Baseline — исторические данные института.

**TTD** (Time To Detection): $t_{\text{alert}} - t_{\text{first signal}}$ — ключевая метрика для новых санкций; регуляторный контекст дедлайнов SAR — в разделе 11.

**Lead time** (self-evolution): $t_{\text{sanction}} - t_{\text{first\_alert}}$ — медианное время от появления в санкционном списке до первого алерта в топ-K (прошедшем DAG-фильтр и cost-based $\tau^*$). Цель: $\le 3$ шагов. Стратификация по эпохам (до/после смены режима). Censored anchors исключаются из медианы, считаются отдельно.

**Latency p99**: 99-й перцентиль времени ответа на $10\,000+$ запросах на production hardware; SLA — CPU-only. Выбор архитектуры (параметры HNSW, квантизация) оценивается trade-off recall@K против латентности (раздел 3).

## 12.5.1. Precision@K панель с bootstrap CI

Операционная панель Precision@K включает:
- Precision@100 / @500 / @1000 с bootstrap 95% CI (percentile method, 600 реплик, seed 72)
- Base rate на каждом временном шаге
- Recall@K на новых санкциях (post-train)
- Cost-based gain vs rule-based baseline: компоненты cost_fp_model, cost_fn_model, cost_baseline, fp_prevented

Доверенные интервалы показывают неопределённость оценки на конечной выборке; CI должен накрывать истинное значение на синтетике.

## 12.5.2. Lead time dashboard

Lead time дашборд показывает:
- Гистограмму lead time (шаги) с медианой и P90
- Стратификацию pre/post смены режима (шаг 43 в Elliptic++)
- Censored count (якори без алерта до санкции)
- Recall@K на post-train санкциях во времени
- Тренд медианного lead time (рост → триггер переобучения)

## 12.6. Метрики стоимости

**Labeling cost** — стоимость верификации разметки. В Spillety якоря (OFAC/EU/UN/OFSI) бесплатны, что обнуляет labeling cost относительно конкурентов с командами разметки (30+ FTE) — это структурное, а не операционное преимущество.

**FTE** — эквивалент полной занятости на обработку алертов; определяется через power analysis и операционные данные (random sampling, Tier 2 review — раздел 9).

**Cost per alert**: $\frac{\text{FTE cost} + \text{infrastructure cost}}{\#\text{alerts}}$ — сводная экономика одного алерта, читается вместе с FP-rate и alert-to-SAR.

## 12.7. Метрики self-evolution

**Median distance** до исторических якорей: $d_{\text{median}}(a_{\text{new}}) = \text{median}_{a\in\mathcal{A}_{\text{old}}}\|z_{a_{\text{new}}}-z_a\|$ с порогом-квантилем — экзамен пространства представлений на новых санкциях.

**Drift KS** — сравнение распределений эмбеддингов на временных срезах; значимый дрейф (по правилу $p+D+d$ из раздела 9) — триггер retraining, мониторинг на dashboard.

**Recall@K на новых санкциях** — доля новых санкций, для которых retrieval возвращает корректные якоря в top-$K$; temporal split обязателен.

## 12.8. Метрики доверия и соответствия

| Метрика | Определение | Измерение |
|---------|-------------|-----------|
| Audit verification | Доля алертов, прошедших Ed25519+Merkle+OTS верификацию | Автоматическая проверка полного контура |
| Merkle proof size | $\log_2 N \cdot 32$ байт | Аналитически + логирование |
| SR 26-2 validation | Покрытие conceptual soundness / outcomes analysis / ongoing monitoring | Чек-лист валидации per tier |
| Bias equalized odds | $\Delta\text{TPR}\approx 0$, $\Delta\text{FPR}\approx 0$ across jurisdictions | Bootstrap CI по группам, SHAP-анализ прокси-признаков |

Выбор fairness-критерия (demographic parity vs equalized odds vs predictive parity) обосновывается bootstrap CI по группам, влиянием на recall@K и латентность, регуляторным контекстом; для Spillety базовый — equalized odds (обоснование — раздел 11). Выборы materiality и подписи/агрегации тестируются соответствующими метриками из этой таблицы.

## 12.9. Что мы не заявляем

Не менее метрик — список заявлений, которые система сознательно не делает:

| Заявление | Статус |
|-----------|--------|
| Абсолютные значения без эмпирической базы | Не заявляются; все метрики с base rate и CI |
| Precision = 1.0 / Recall = 1.0 / Zero FP | Недостижимо при дисбалансе и неполной разметке |
| Distance = causal | Неверно; расстояние — корреляция в learned space |
| Court-admissibility без Daubert | Требует внешней валидации (error rate, peer review) |

Основания: неполнота ground truth (Elliptic++: unknown = 557k), зависимость precision от base rate (Bitcoin против Ethereum, CEX против DEX), темпоральный дрейф, стоимость сбора TTD/alert-to-SAR, допущения нормальной аппроксимации в power analysis, непубличность метрик конкурентов.

## 12.10. Power analysis для random sampling

Оценка precision с допуском $\pm e$ при уровне $1-\alpha$:

$$
n = \frac{z_{1-\alpha/2}^2 \cdot p(1-p)}{e^2}
$$

Пример: $p=0.05$, $e=0.01$, 95% CI → $n \approx 1825$. Размер random sampling из auto-clear пересчитывается ежемесячно; power analysis обеспечивает unbiased оценку precision/recall (механика — раздел 9).

## 12.11. Визуализация

Код генерации графиков: [`files/11_visualization.py`](./files/11_visualization.py).

![pr auc comparison](./files/11-10-1_pr_auc_comparison.png)

**PR-AUC comparison:** кривые baseline vs GBDT vs GBDT+calibration; горизонтальная линия — base rate.

![calibration comparison](./files/11-10-2_calibration_comparison.png)

**Calibration comparison:** reliability diagram и ECE; выбор isotonic vs beta по ECE.

![power analysis](./files/11-10-3_power_analysis.png)

**Power analysis:** зависимость $n(p)$ при фиксированном $e$.

![drift monitoring](./files/11-10-4_drift_monitoring.png)

**Drift monitoring:** KS-статистика во времени; превышение порога — триггер retraining.

**Lead time:** гистограмма lead time (шаги) с медианой и P90, стратификация pre/post regime change (шаг 43 в Elliptic++), censored count (якори без алерта до санкции), Recall@K на post-train санкциях во времени. Рост медианного lead time → триггер переобучения.

Дополнительно: bootstrap CI для PR-AUC/Precision@K/Recall@K и ECE/Brier, гистограммы latency p99, Merkle proof size против $N$.

## 12.12. Ограничения контура

1. Неполнота разметки — все precision/recall суть нижние оценки.
2. Зависимость от base rate — сравнение без нормализации бессмысленно.
3. Темпоральный дрейф — метрики требуют периодического пересчёта.
4. TTD/alert-to-SAR появляются только после прод-запуска.
5. Power analysis чувствителен к априорной оценке $p$.
6. Сравнение с конкурентами — только по косвенным данным (FTE, latency, cost).

---

## См. также

- [12_metrics_dashboard.ipynb](../notebooks/12_metrics_dashboard.ipynb) — дашборд метрик и мониторинг качества.
- [13_end_to_end_pipeline.ipynb](../notebooks/13_end_to_end_pipeline.ipynb) — end-to-end пайплайн с полным контуром метрик.