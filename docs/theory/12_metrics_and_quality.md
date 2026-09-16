# 12. Метрики и оценка качества

### Термины

| Термин | Определение |
|--------|-------------|
| PR-AUC | Площадь под кривой precision-recall |
| Precision@K | Доля релевантных объектов среди top-$K$ возвратов |
| Recall@K | Доля истинных соседей, попавших в top-$K$ |
| Brier score | Средний квадрат отклонения предсказанной вероятности от исхода |
| ECE | Expected Calibration Error — ожидаемая ошибка калибровки |
| TTD | Time To Detection — время от первого on-chain сигнала до алерта |
| FP-rate | Частота ложных срабатываний на аналитика |
| alert-to-SAR | Доля алертов, конвертируемых в SAR |
| Latency p99 | 99-й перцентиль времени ответа |
| Cost per alert | Полная стоимость обработки одного алерта |
| Drift KS | Статистика и p-value теста Колмогорова–Смирнова для сдвига распределений |
| Audit verification | Доля успешных криптографических верификаций алерта |
| Merkle proof size | Размер доказательства включения в Merkle-дерево (байт) |
| SR 26-2 validation | Полнота валидации по SR 26-2 (conceptual soundness / outcomes analysis / ongoing monitoring) |
| Bias equalized odds | Равенство TPR и FPR across groups |
| FTE | Full-Time Equivalent — эквивалент полной занятости |
| Power analysis | Расчёт минимального размера выборки для заданной мощности теста |
| Base rate | Доля положительного класса в популяции |
| Walk-forward validation | Последовательный сдвиг train/test окон без утечки будущего |

---

## 12.1. Постановка задачи

Компоненты Spillety (entity resolution, contrastive learning, causal-фильтр, GBDT, калибровка, cost-функция, temporal validation, evidence/WORM) обладают локальными метриками. Система в целом требует единого метрического контура, который:

1. Отражает фактические характеристики без завышенных заявлений.
2. Сравним с baseline (тривиальный предиктор, XGBoost на Elliptic++).
3. Оценён на hold-out с временной изоляцией.
4. Разделяет retrieval, scoring, калибровку, операционный и стоимостный контуры, а также self-evolution.
5. Явно учитывает base rate.

> Теоретически мы оцениваем наше решение вот так: PR-AUC, Precision@K, Recall@K, Brier, ECE, FP-rate, alert-to-SAR, TTD, latency p99, cost per alert, drift KS, audit verification, Merkle proof size, SR 26-2 validation, bias equalized odds.

---

## 12.2. Таксономия метрик

| Группа | Объект измерения | Метрики |
|--------|------------------|---------|
| Качество ранжирования | Разделение illicit/legit | PR-AUC, Precision@K, Recall@K |
| Калибровка | Соответствие вероятностей частотам | Brier, ECE |
| Операционные | Поведение в проде | FP-rate на аналитика, alert-to-SAR, TTD, latency p99 |
| Стоимость | Экономика | labeling cost, FTE, cost per alert |
| Self-evolution | Адаптивность | temporal validation (median distance), drift KS, recall@K на новых санкциях |
| Доверие и соответствие | Проверяемость | audit verification, Merkle proof size, SR 26-2 validation, bias equalized odds |

---

## 12.3. Метрики качества ранжирования

### 12.3.1. PR-AUC

\[
\text{PR-AUC} = \int_0^1 \text{Precision}(\text{Recall}) \, d\text{Recall}
\]

В отличие от ROC-AUC, чувствителен к base rate. При низкой доле положительного класса высокий ROC-AUC может маскировать низкий precision. Оценка — hold-out temporal split (train на прошлом, test на будущем), кривая на test. Baseline — XGBoost на Elliptic++ (56 признаков). PR-AUC отражает потенциал модели, но не заменяет precision при рабочем пороге.

### 12.3.2. Precision@K

\[
\text{Precision@K} = \frac{\|\{\text{relevant}\} \cap \{\text{top-}K\}\|}{K}
\]

Для retrieval: доля истинно связанных anchors среди top-$K$. Параметр $K$ выбирается по cost-функции, отчёт сопровождается base rate.

### 12.3.3. Recall@K

\[
\text{Recall@K} = \frac{\|\{\text{relevant}\} \cap \{\text{top-}K\}\|}{\|\{\text{relevant}\}\|}
\]

Оценка на уровне кластеров (более стабильная единица, блок 5). Temporal-валидация: recall@K на санкциях, добавленных после даты обучения, демонстрирует обобщение на unseen anchors. Доверительные интервалы — bootstrap CI.

**Выбор архитектуры retrieval** оценивается по recall@K и latency p99 с bootstrap CI.

---

## 12.4. Метрики калибровки

### 12.4.1. Brier score

\[
\text{Brier} = \frac{1}{n}\sum_{i=1}^{n}(\hat{p}_i - y_i)^2
\]

Интерпретация: $0$ — идеально, $0.25$ — случайный предиктор при сбалансированных классах (для imbalanced — смещён). Baseline — тривиальный предиктор $\hat{p}_i \equiv \text{base rate}$.

### 12.4.2. ECE

\[
\text{ECE} = \sum_{m=1}^{M}\frac{|B_m|}{n}\left|\text{acc}(B_m) - \text{conf}(B_m)\right|
\]

$B_m$ — бин $m$, $M=10$–$15$. Порог плохой калибровки: $>0.1$.

### 12.4.3. Выбор метода калибровки: isotonic vs beta

| Вариант | Предположения | Требование к данным |
|---------|---------------|---------------------|
| Isotonic regression | Непараметрическая монотонность | Большой hold-out |
| Beta calibration | Параметрическая форма | Устойчива на малых выборках |

**Решение между isotonic vs beta принимается по ECE/Brier на hold-out temporal split:** критерий — минимальный ECE при стабильном Brier, проверка — bootstrap CI для разности ECE, а также сохранение recall@K при фиксированном ECE. Надёжность оценивается reliability diagram.

---

## 12.5. Операционные метрики

### 12.5.1. FP-rate на аналитика

\[
\text{FP-rate} = \frac{\text{FP}}{\text{TP}+\text{FP}} \quad \text{per analyst per unit time}
\]

Источник — операционные данные с разметкой аналитика. Связан с cost-функцией через $C_{FP}$: рост FP-rate сдвигает оптимальный порог вправо.

### 12.5.2. Alert-to-SAR conversion

\[
\text{alert-to-SAR} = \frac{\#\text{SAR}}{\#\text{alerts}}
\]

Отражает полезность алертов для compliance; низкий уровень — индикатор шума. Baseline — исторические данные.

### 12.5.3. TTD

\[
\text{TTD} = t_{\text{alert}} - t_{\text{first signal}}
\]

Ключевая метрика для novel sanctions.

### 12.5.4. Latency p99

99-й перцентиль времени ответа на $10\,000+$ запросов на production hardware. SLA — CPU-only, интерактивная задержка; p99 сопровождается bootstrap CI. Выбор между вариантами архитектуры (HNSW-параметры, продуктовая квантизация) оценивается trade-off recall@K vs latency.

---

## 12.6. Метрики стоимости

### 12.6.1. Labeling cost

Стоимость верификации разметки (в Spillety — court documents). Anchors (OFAC/EU/UN/OFSI) бесплатны, что обнуляет labeling cost относительно конкурентов (30+ FTE).

### 12.6.2. FTE

Определяется через power analysis и операционные данные (random sampling, Tier 2 review).

### 12.6.3. Cost per alert

\[
\text{Cost per alert} = \frac{\text{FTE cost} + \text{infrastructure cost}}{\#\text{alerts}}
\]

> Теоретически мы оцениваем стоимость вот так: cost per alert совместно с latency p99, FP-rate и alert-to-SAR, а также PR-AUC, Precision@K, Recall@K, Brier, ECE, TTD, drift KS, audit verification, Merkle proof size, SR 26-2 validation, bias equalized odds.

---

## 12.7. Метрики self-evolution

### 12.7.1. Temporal validation

Median distance до исторических anchor→anchor пар:

$$d_{\text{median}}(a_{\text{new}}) = \text{median}_{a\in\mathcal{A}_{\text{old}}}\|z_{a_{\text{new}}}-z_a\|.$$

Порог — квантиль распределения исторических $d_{\text{median}}$.

### 12.7.2. Drift (KS)

Сравнение распределений embeddings на временных срезах; $p<0.05$ — значимый дрейф, триггер retraining. Мониторинг — dashboard KS-статистики.

### 12.7.3. Recall@K на новых санкциях

Доля новых санкций, для которых retrieval возвращает корректные anchors в top-$K$. Temporal split обязателен.

---

## 12.8. Метрики доверия и соответствия

| Метрика | Определение | Как измеряется |
|---------|-------------|----------------|
| Audit verification | Доля алертов, прошедших Ed25519+Merkle+OTS верификацию | Автоматическая проверка полного контура |
| Merkle proof size | $\log_2 N \cdot 32$ байт | Аналитически + логирование |
| SR 26-2 validation | Покрытие conceptual soundness / outcomes analysis / ongoing monitoring | Чек-лист валидации per tier (High/Medium/Low) |
| Bias equalized odds | $\Delta\text{TPR}\approx 0$, $\Delta\text{FPR}\approx 0$ across jurisdictions | Bootstrap CI по группам, SHAP-анализ прокси-признаков |

**Выбор fairness-критерия: demographic parity vs equalized odds vs predictive parity.** Между вариантами выбор обосновывается bootstrap CI по группам, влиянием на recall@K и latency, а также регуляторным контекстом. Для Spillety базовым принят equalized odds.

**Выбор materiality: High vs Medium vs Low** тестируется SR 26-2 validation, PR-AUC/Precision@K/Recall@K с bootstrap CI и latency p99.

**Выбор подписи/агрегации: Ed25519 vs ECDSA, Merkle vs flat** тестируется latency, audit verification и Merkle proof size (bootstrap CI, recall@K как контроль деградации retrieval при квантизации).

---

## 12.9. Ограничения и заявления

| Заявление | Статус |
|-----------|--------|
| Абсолютные значения без эмпирической базы | Не заявляются; все метрики сопровождаются base rate и CI |
| Precision = 1.0 / Recall = 1.0 / Zero FP | Недостижимо при imbalanced и неполной разметке |
| Distance = causal | Неверно; distance — корреляция в learned space |
| Court-admissibility без Daubert | Требует внешней валидации (error rate, peer review) |

Основания: неполнота ground truth (Elliptic++: unknown=557k), зависимость precision от base rate (Bitcoin vs Ethereum, CEX vs DEX), temporal drift, стоимость сбора TTD/alert-to-SAR, допущения нормальной аппроксимации в power analysis, непубличность метрик конкурентов.

---

## 12.10. Power analysis для random sampling

Оценка precision с допуском $\pm e$ при $1-\alpha$:

\[
n = \frac{z_{1-\alpha/2}^2 \cdot p(1-p)}{e^2}
\]

Пример: $p=0.05$, $e=0.01$, $95\%$ CI $\to n\approx 1825$. Для Spillety размер random sampling из auto-clear пересчитывается ежемесячно; power analysis обеспечивает unbiased оценку precision/recall (ср. блок 8, active learning без selection bias).

---

## 12.11. Визуализация

> Код для генерации графиков вынесен в [`files/11_visualization.py`](./files/11_visualization.py) — сохранён без изменений.

![pr auc comparison](./files/11-10-1_pr_auc_comparison.png)

**PR-AUC comparison:** PR-кривые baseline vs GBDT vs GBDT+calibration; горизонтальная линия — base rate.

![calibration comparison](./files/11-10-2_calibration_comparison.png)

**Calibration comparison:** reliability diagram и ECE; выбор isotonic vs beta по ECE.

![power analysis](./files/11-10-3_power_analysis.png)

**Power analysis:** зависимость $n(p)$ для фиксированного $e$.

![drift monitoring](./files/11-10-4_drift_monitoring.png)

**Drift monitoring:** KS-статистика во времени; превышение порога — триггер retraining.

Дополнительно: bootstrap CI для PR-AUC/Precision@K/Recall@K и ECE/Brier, latency-гистограммы p99, Merkle proof size vs $N$.

---

## 12.12. Сводка архитектурных выборов

| Выбор | Альтернативы | Тесты |
|-------|--------------|-------|
| Калибровка | isotonic vs beta | ECE, Brier на hold-out, bootstrap CI |
| Подпись Evidence JSON | Ed25519 vs ECDSA | latency p99, audit verification, bootstrap CI |
| Агрегация | Merkle vs flat | Merkle proof size, audit verification |
| Materiality | High vs Medium vs Low | SR 26-2 validation, recall@K, latency |
| Fairness | demographic parity vs equalized odds vs predictive parity | equalized odds, bootstrap CI, FP-rate |

Каждый выбор документируется протоколом: temporal split, walk-forward, bootstrap CI, reliability diagram, latency-бенчмарк.

---

## 12.13. Ограничения метрического контура

1. Неполнота разметки — нижняя оценка precision/recall.
2. Зависимость от base rate — несравнимость без нормализации.
3. Temporal drift — необходимость периодического пересчёта.
4. TTD/alert-to-SAR доступны только после прод-запуска.
5. Power analysis — чувствительность к оценке $p$.
6. Сравнение с конкурентами — по косвенным данным (FTE, latency, cost).

---

**Резюме.** Метрический контур Spillety охватывает ранжирование, калибровку, операционные и стоимостные показатели, self-evolution и доверие.

| Группа | Ключевые метрики | Протокол |
|--------|------------------|----------|
| Ранжирование | PR-AUC, Precision@K, Recall@K | Hold-out temporal split, bootstrap CI |
| Калибровка | Brier, ECE (isotonic vs beta) | Reliability diagram, ECE/Brier CI |
| Операционные | FP-rate, alert-to-SAR, TTD, latency p99 | Операционные данные, бенчмарк |
| Стоимость | labeling cost, FTE, cost per alert | Операционные данные, power analysis |
| Self-evolution | temporal validation, drift KS, recall@K новых санкций | Мониторинг, temporal split |
| Доверие | audit verification, Merkle proof size, SR 26-2, equalized odds | Криптоверификация, чек-лист, bootstrap CI |

> Теоретически мы оцениваем наше решение вот так: PR-AUC, Precision@K, Recall@K, Brier, ECE, FP-rate, alert-to-SAR, TTD, latency p99, cost per alert, drift KS, audit verification, Merkle proof size, SR 26-2 validation, bias equalized odds — полный набор, измеряемый на temporal hold-out с bootstrap CI и бенчмарками latency/recall.

**Практические выводы:**

- PR-AUC дополняется precision при рабочем пороге с указанием base rate.
- Brier и ECE взаимодополняют друг друга; выбор isotonic vs beta — по ECE/Brier на hold-out с bootstrap CI.
- FP-rate связан с cost-функцией и порогом.
- Labeling cost и FTE — ключевые для unit economics.
- Temporal validation и drift KS — основа self-evolution.
- Random sampling требует power analysis.
- Audit verification, Merkle proof size, SR 26-2 validation и equalized odds — обязательная часть отчётности.
