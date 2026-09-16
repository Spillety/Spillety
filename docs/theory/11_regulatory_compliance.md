# 11. Регуляторное соответствие: SR 26-2, FATF, FinCEN, 6AMLD, Daubert и bias-аудит

### Термины

| Термин | Определение |
|--------|-------------|
| SR 26-2 | Revised Guidance on Model Risk Management, Federal Reserve, 17 апреля 2026; заменяет SR 11-7 и SR 21-8 |
| Model Risk Management (MRM) | Управление рисками, связанными с моделями: governance, валидация, мониторинг |
| Materiality | Произведение model exposure (значимость выхода для решений) и model purpose (регуляторное/финансовое назначение) |
| FATF Rec. 16 (Travel Rule) | Требование сопровождения платежа информацией об отправителе и получателе; для virtual assets — off-chain обмен между VASP |
| FinCEN SAR | Suspicious Activity Report, подаваемый при подозрении на illicit-активность |
| 6AMLD | Sixth Anti-Money Laundering Directive ЕС: гармонизация predicate offences |
| Daubert | Критерии допустимости экспертных показаний в суде США: testability, error rate, peer review, general acceptance |
| FFIEC | Federal Financial Institutions Examination Council; стандарты BSA/AML-экзамена |
| Bias audit | Проверка систематического смещения модели относительно защищённых групп |
| Equalized odds | Равенство TPR и FPR across groups |

---

## 11.1. Постановка задачи

Техническая корректность системы не тождественна регуляторной приемлемости. Финансовый институт не вправе эксплуатировать инструмент, не классифицированный по SR 26-2, не соответствующий FATF Rec. 16, не генерирующий FinCEN SAR и не удовлетворяющий критериям Daubert.

Spillety рассматривается в пяти регуляторных режимах:

1. **SR 26-2** — классификация компонентов, materiality, валидация и governance.
2. **FATF Rec. 16** — Travel Rule для VASP-транзакций.
3. **FinCEN SAR** — генерация и своевременная подача отчётов.
4. **6AMLD** — учёт predicate offences в приоритизации алертов.
5. **Daubert** — обеспечение court-admissibility.

Оценка: PR-AUC, Precision@K, Recall@K, Brier, ECE, FP-rate, alert-to-SAR, TTD, latency p99, cost per alert, drift KS, audit verification, Merkle proof size, SR 26-2 validation, bias equalized odds.

---

## 11.2. SR 26-2: классификация и materiality

### 11.2.1. Отличия от SR 11-7

| Область | SR 11-7 | SR 26-2 |
|---------|---------|---------|
| Governance | Единообразная строгость | Risk-based, tiered by materiality |
| Определение модели | Широкое, включало rule-based tools | Сужено до сложных количественных методов со статистической/экономической/финансовой теорией |
| Materiality | Имплицитная | Явная: exposure × purpose |
| GenAI / Agentic AI | Не адресовано | Явно вне scope; принципы применяются по аналогии |
| Последствия несоответствия | Возможна supervisory criticism | Non-compliance с guidance alone не влечёт criticism; action возможен при unsafe or unsound practices |
| Мониторинг | Validation-centric | Усилен ongoing monitoring и outcomes analysis |

SR 26-2 применяется к banking organizations с активами > $30 млрд. Руководство не устанавливает enforceable standards, но служит основанием для оценки soundness.

### 11.2.2. Модель materiality

Materiality — сочетание:

- **Model exposure** — влияние выхода модели на бизнес-решения.
- **Model purpose** — использование для регуляторных требований или финансового риск-менеджмента.

| Tier | Exposure | Purpose | Режим governance |
|------|----------|---------|------------------|
| High | Критичный для решений | Регуляторный | Полная валидация, независимый обзор |
| Medium | Влияет на решения | Финансовый | Валидация + мониторинг |
| Low | Информационный | Внутренний | Идентификация + performance monitoring |

**Выбор уровня materiality: High vs Medium vs Low.** Решение принимается на тестах: bootstrap CI для PR-AUC и Precision@K/Recall@K на temporal split, latency p99 бенчмарк, SR 26-2 validation outcomes, оценка TTD и alert-to-SAR. High-tier требует annual independent validation; Medium — validation + monitoring; Low — ongoing monitoring.

### 11.2.3. Классификация компонентов Spillety

| Компонент | Классификация по SR 26-2 | Основание |
|-----------|--------------------------|-----------|
| Contrastive encoder (GraphSAGE) | **Model** | Сложный количественный метод с теорией contrastive learning |
| Causal DAG | **Expert-based model** | Документированные допущения; не статистический метод в строгом смысле |
| GBDT (LightGBM) | **Model** | Количественный метод со статистической теорией |
| Rule-based filters | **Может не квалифицироваться как model** | Детерминированные правила без статистической теории |
| HNSW retrieval | **Not model** | Детерминированный алгоритм поиска |
| LLM Explainer | **Исключён из scope** | Generative AI вне SR 26-2 (принципы governance — по аналогии) |

Независимо от классификации non-model компоненты подлежат FFIEC independent testing: верификация логики и реализации.

### 11.2.4. Требования к валидации

Три обязательные области:

1. **Conceptual soundness** — соответствие дизайна риск-профилю института.
2. **Outcomes analysis** — above-the-line и below-the-line тестирование.
3. **Ongoing monitoring** — детекция drift и деградации.

Дополнение для AML: **data integrity testing** — верификация поступления данных, необходимых модели. Пример enforcement: штраф Wise US $4.2M (июль 2025) за SAR deficiencies и data integrity failures.

**Above-the-line / below-the-line:**

- **Below-the-line:** снижение порога ниже production, replay исторических транзакций, обзор сработавших алертов — оценка under-detection.
- **Above-the-line:** повышение порога, сэмплирование потерянных алертов — оценка продуктивности потерянных срабатываний.

Методология adjustments, расчёты размера выборки и результаты документируются; смещения оцениваются bootstrap CI.

### 11.2.5. Частота валидации

Фиксированная частота не установлена; режим — risk-based, зависящий от materiality, скорости изменений и ограничений данных. Для Spillety: high-materiality (GBDT в decision path) — annual validation; low-materiality (retrieval) — ongoing monitoring и периодический review. Дрейф контролируется KS-тестом и мониторингом Brier/ECE.

---

## 11.3. FATF Rec. 16 (Travel Rule)

### 11.3.1. Нормативные требования

Информация об отправителе и получателе обязана сопровождать платёжное сообщение при cross-border переводе; для virtual assets — передача между VASP off-chain по защищённым каналам.

**Стандартизированные поля (суммы > USD/EUR 1 000):**

| Поле | Требуется |
|------|-----------|
| Имя отправителя | Да |
| Имя получателя | Да |
| Адрес или country/town (originator) | Да |
| Дата рождения (originator) | Да |
| Account number или unique transaction reference | Да |

**Ревизия Rec. 16 (июнь 2025):** уточнена ответственность в payment chain, стандартизированы требования, введено обязательство внедрять средства защиты от fraud/error, срок имплементации — конец 2030 г.

**Пороги по юрисдикциям:**

| Юрисдикция | Порог |
|------------|-------|
| FATF default | USD/EUR 1 000 |
| US FinCEN | USD 3 000 |
| EU (TFR 2023/1113) | €0 (без порога) |
| UK | £1 000 |
| Canada | CAD 1 000 |

ЕС — outlier: нулевой порог для CASP-to-CASP, верификация self-hosted wallet при ≥ €1 000.

### 11.3.2. Соответствие Spillety

Evidence JSON включает originator/beneficiary-поля для VASP-транзакций; off-chain данные — через VASP-интеграции (фаза 2). Проблема **Sunrise Issue** (контрагент в юрисдикции без имплементации Travel Rule) решается enhanced due diligence, запросом данных у клиента напрямую и ограничением активности с high-risk регионами.

---

## 11.4. FinCEN SAR

### 11.4.1. Режим подачи

- **Срок:** 30 календарных дней после initial detection; дополнительно 30 дней для идентификации suspect, но не более 60 дней суммарно.
- **Порог:** $5 000 (банки), $2 000 (MSB).
- **Триггеры:** средства от нелегальной деятельности; структурирование для обхода отчётности; отсутствие деловой/законной цели; содействие преступной деятельности.
- **Continuing activity** (FAQ октябрь 2025): review каждые 90 дней, continuing SAR в течение 30 дней после review — суммарно 120 дней.

### 11.4.2. Генерация SAR из Evidence JSON

| Поле SAR | Источник |
|----------|----------|
| Transaction hash | on-chain |
| Blockchain | on-chain |
| Timestamp | on-chain |
| Sender / Receiver | on-chain |
| Amount (crypto) | on-chain |
| Amount (USD) | price oracle |
| Causal path | Evidence JSON |
| Anchor provenance | Evidence JSON |
| Risk score | GBDT (калиброванный) |

SAR не подаётся автоматически; обязателен human review аналитика.

---

## 11.5. 6AMLD

6AMLD гармонизирует 22 predicate offences (включая cybercrime, environmental crime, tax crime), криминализует aiding/abetting/inciting/attempting, вводит ответственность юридических лиц и минимальный срок 4 года, требует dual criminality.

**Соответствие:** DAG учитывает predicate offence в приоритизации; метаданные anchor включают тип predicate offence (из судебных документов); признаки GBDT содержат `predicate_offence_severity`.

---

## 11.6. Daubert court-admissibility

### 11.6.1. Критерии

| Критерий | Проверка |
|----------|----------|
| Testability | Верифицируемость методологии |
| Error rate | Известная частота ошибок |
| Peer review | Внешняя проверка |
| General acceptance | Принятие в relevant field |

Фокус inquiry — принципы и методология, а не выводы.

### 11.6.2. Соответствие Spillety

| Критерий | Реализация |
|----------|------------|
| Testability | PR-AUC, Brier, ECE на held-out temporal split |
| Error rate | Precision@K, Recall@K, FP-rate per analyst, bootstrap CI |
| Peer review | Independent validation внешней командой |
| General acceptance | Contrastive learning + GBDT — устоявшиеся методы |

Без внешней Daubert-валидации заявление о court-admissibility недопустимо. Требуется adversarial testing (red team ежемесячно) на устойчивость к adversarial anchors.

---

## 11.7. Bias-аудит

### 11.7.1. Постановка

Регуляторы (EBA, AI Act) требуют детекции и митигации нежелательного смещения. В FCP-домене риски: повышенный FP-rate для отдельных юрисдикций, country-contingent differential treatment, levelling down при попытке выравнивания precision.

### 11.7.2. Метрики справедливости

| Метрика | Определение |
|---------|-------------|
| Demographic parity | Равенство positive rate across groups |
| Equalized odds | Равенство TPR и FPR across groups |
| Predictive parity | Равенство precision (PPV) across groups |

Метрики несовместимы одновременно; выбор зависит от контекста. Для юрисдикционного аудита Spillety приоритет — **equalized odds** (контроль TPR и FPR), поскольку регулятору критичны как пропуски, так и ложные срабатывания. Оценка — bootstrap CI по группам, проверка significance, отчёт по FP-rate и alert-to-SAR.

### 11.7.3. Jurisdiction-level fairness audit

1. Стратификация алертов по юрисдикциям.
2. Оценка precision, recall, FPR per group.
3. Проверка equalized odds: близость TPR и FPR across jurisdictions с bootstrap CI.
4. При значимой диспропорции — feature-level investigation.

Диспропорция может быть **justified** (концентрация OFAC-санкций) или **unjustified**; различие требует экспертизы и документации.

### 11.7.4. SHAP для детекции скрытого смещения

Признак, коррелирующий с юрисдикцией (например, `news_co_mention_count`), может выступать прокси для sensitive attribute. SHAP-анализ выявляет такие признаки и направляет митигацию (регуляризация, исключение, перевзвешивание).

### 11.7.5. Выбор и тестирование fairness-критерия

Между demographic parity, equalized odds и predictive parity выбор обосновывается на тестах: bootstrap CI для диспропорции, recall@K per group, latency overhead при коррекции. Equalized odds выбран как основной для bias equalized odds в метрическом контуре.

---

## 11.8. Архитектурные выборы и их проверка

| Выбор | Альтернативы | Тесты |
|-------|--------------|-------|
| Подпись Evidence JSON | Ed25519 vs ECDSA | latency p99, bootstrap CI, audit verification |
| Агрегация алертов | Merkle vs flat | Merkle proof size, верификация $O(\log N)$, bootstrap CI |
| Калибровка скора (влияет на SAR-порог) | isotonic vs beta | ECE, Brier на hold-out, bootstrap CI |
| Materiality | High vs Medium vs Low | SR 26-2 validation, TTD, recall@K, alert-to-SAR |
| Fairness | demographic parity vs equalized odds vs predictive parity | bootstrap CI, FP-rate, recall@K, equalized odds |

---

## 11.9. Визуализация

![materiality matrix](./files/10-8-1_materiality_matrix.png)

**Materiality matrix:** exposure × purpose. Красная зона — high materiality (GBDT, encoder), зелёная — low (HNSW, правила).

![fairness audit](./files/10-8-2_fairness_audit.png)

**Fairness audit:** TPR, FPR, Precision по юрисдикциям; equalized odds требует близости TPR/FPR.

![sar timeline](./files/10-8-3_sar_timeline.png)

**SAR timeline:** detection → review → escalation → filing (30/60 дней) → continuing SAR (120 дней).

---

## 11.10. Метрический контур регуляторного соответствия

Оценка: PR-AUC, Precision@K, Recall@K, Brier, ECE, FP-rate, alert-to-SAR, TTD, latency p99, cost per alert, drift KS, audit verification, Merkle proof size, SR 26-2 validation, bias equalized odds.

SR 26-2 validation измеряется как полнота conceptual soundness / outcomes analysis / ongoing monitoring; bias — equalized odds across jurisdictions; audit — верификация Ed25519+Merkle+OTS.

---

## 11.11. Ограничения

1. SR 26-2 — guidance, а не enforceable standard; non-compliance alone не влечёт criticism, но unsafe practices влекут action.
2. Оценка materiality субъективна; критерии exposure/purpose определяются институтом.
3. GenAI вне scope SR 26-2, но принципы governance применимы к LLM Explainer.
4. FATF Rec. 16 имплементирован неравномерно; Sunrise Issue — операционный риск.
5. FinCEN SAR дедлайны жёсткие; auto-block и TTD должны обеспечивать запас.
6. 6AMLD dual criminality усложняет cross-jurisdiction приоритизацию.
7. Daubert требует adversarial testing; без него court-admissibility под вопросом.
8. Различение justified vs unjustified disparity при bias-аудите не формализовано.
9. Метрики справедливости несовместимы одновременно.

---

## См. также

- [12_metrics_dashboard.ipynb](../notebooks/12_metrics_dashboard.ipynb) — метрики качества и мониторинг compliance.
