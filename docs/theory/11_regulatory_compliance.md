# 11. Регуляторное соответствие: SR 26-2, FATF, FinCEN, 6AMLD, Daubert и bias-аудит

Техническая корректность системы и её регуляторная приемлемость — разные вещи, и вторая не следует из первой автоматически. Финансовый институт не вправе эксплуатировать инструмент, не классифицированный по SR 26-2, не соответствующий FATF Rec. 16, не генерирующий FinCEN SAR и не удовлетворяющий критериям Daubert. Эта глава разбирает пять регуляторных режимов, в которых живёт Spillety, и показывает, какие свойства системы каким требованиям отвечают.

## 11.1. SR 26-2: классификация и materiality

### 11.1.1. Что изменилось относительно SR 11-7

| Область | SR 11-7 | SR 26-2 |
|---------|---------|---------|
| Governance | Единообразная строгость | Risk-based, tiered by materiality |
| Определение модели | Широкое, включало rule-based tools | Сужено до сложных количественных методов со статистической/экономической/финансовой теорией |
| Materiality | Имплицитная | Явная: exposure × purpose |
| GenAI / Agentic AI | Не адресовано | Явно вне scope; принципы применяются по аналогии |
| Последствия несоответствия | Возможна supervisory criticism | Non-compliance с guidance alone не влечёт criticism; action возможен при unsafe or unsound practices |
| Мониторинг | Validation-centric | Усилен ongoing monitoring и outcomes analysis |

SR 26-2 (Revised Guidance on Model Risk Management, Federal Reserve) применяется к banking organizations с активами более $30 млрд; руководство не устанавливает enforceable standards, но служит основанием оценки soundness. Для Spillety важно следствие из суженного определения модели: детерминированные компоненты без статистической теории могут не квалифицироваться как model — но это не освобождает их от проверки.

### 11.1.2. Materiality

Materiality — сочетание **model exposure** (влияние выхода модели на решения) и **model purpose** (регуляторное или финансовое назначение):

| Tier | Exposure | Purpose | Режим governance |
|------|----------|---------|------------------|
| High | Критичный для решений | Регуляторный | Полная валидация, независимый обзор |
| Medium | Влияет на решения | Финансовый | Валидация + мониторинг |
| Low | Информационный | Внутренний | Идентификация + performance monitoring |

Уровень для Spillety — High (GBDT в decision path определяет блокировки и SAR), и это не ярлык, а обязательства: annual independent validation, outcomes analysis, ongoing monitoring. Выбор уровня проверяется тестами: bootstrap CI для PR-AUC и Precision@K/Recall@K на temporal split, latency p99, полнота SR 26-2 validation, TTD и alert-to-SAR.

### 11.1.3. Классификация компонентов

| Компонент | Классификация | Основание |
|-----------|---------------|-----------|
| Contrastive encoder (GraphSAGE) | **Model** | Сложный количественный метод с теорией contrastive learning |
| Causal DAG | **Expert-based model** | Документированные допущения; не статистический метод в строгом смысле |
| GBDT (LightGBM) | **Model** | Количественный метод со статистической теорией |
| Rule-based filters | **Может не квалифицироваться как model** | Детерминированные правила без статистической теории |
| HNSW retrieval | **Not model** | Детерминированный алгоритм поиска |
| LLM Explainer | **Вне scope** | Generative AI вне SR 26-2 (принципы governance — по аналогии) |

Независимо от классификации non-model компоненты подлежат FFIEC independent testing — верификации логики и реализации.

### 11.1.4. Валидация

Три обязательные области SR 26-2: **conceptual soundness** (соответствие дизайна риск-профилю), **outcomes analysis** (above-the-line и below-the-line тестирование), **ongoing monitoring** (детекция дрейфа). Для AML добавляется **data integrity testing** — верификация поступления данных, необходимых модели; показателен штраф Wise US $4.2M (июль 2025) именно за SAR deficiencies и data integrity failures.

Above-the-line / below-the-line тестирование — практический инструмент outcomes analysis. Below-the-line: порог опускается ниже production, исторические транзакции реплеятся, сработавшие алерты разбираются — так оценивается under-detection. Above-the-line: порог поднимается, из потерянных алертов берётся выборка — оценивается, была ли в них продуктивность. Методология, размеры выборок и результаты документируются; смещения оцениваются bootstrap CI.

Частота валидации фиксированной не является — режим risk-based по materiality и скорости изменений. Для Spillety: high-materiality GBDT — annual validation; low-materiality retrieval — ongoing monitoring с периодическим review. Дрейф контролируется KS-тестом и мониторингом Brier/ECE (раздел 9).

## 11.2. FATF Rec. 16 (Travel Rule)

### 11.2.1. Требования

Информация об отправителе и получателе обязана сопровождать платёж при cross-border переводе; для виртуальных активов — передаваться между VASP off-chain по защищённым каналам. Стандартизированные поля (для сумм свыше USD/EUR 1 000): имя отправителя, имя получателя, адрес или country/town originator, дата рождения originator, номер счёта или уникальная ссылка транзакции.

Ревизия Rec. 16 (июнь 2025) уточнила ответственность в payment chain, стандартизировала требования, ввела обязательство средств защиты от fraud/error со сроком имплементации до конца 2030 года. Пороги по юрисдикциям различаются:

| Юрисдикция | Порог |
|------------|-------|
| FATF default | USD/EUR 1 000 |
| US FinCEN | USD 3 000 |
| EU (TFR 2023/1113) | €0 (без порога) |
| UK | £1 000 |
| Canada | CAD 1 000 |

ЕС — выброс: нулевой порог для CASP-to-CASP, верификация self-hosted wallet при ≥ €1 000.

### 11.2.2. Соответствие Spillety

Evidence JSON включает originator/beneficiary-поля для VASP-транзакций; off-chain данные — через VASP-интеграции (фаза 2). Проблема **Sunrise Issue** — контрагент в юрисдикции без имплементации Travel Rule — решается enhanced due diligence, запросом данных у клиента напрямую и ограничением активности с high-risk регионами.

## 11.3. FinCEN SAR

Режим подачи жёсткий по срокам: 30 календарных дней после initial detection, ещё 30 дней на идентификацию suspect, суммарно не более 60; пороги — $5 000 для банков, $2 000 для MSB. Триггеры: средства от нелегальной деятельности, структурирование для обхода отчётности, отсутствие деловой или законной цели, содействие преступной деятельности. Для continuing activity (FAQ октябрь 2025): review каждые 90 дней, continuing SAR в течение 30 дней после review — до 120 дней суммарно.

Генерация SAR из Evidence JSON описана в разделе 10.4: поля, источники, обязательный human review. Здесь важно регуляторное следствие для операционной точки: жёсткие дедлайны означают, что auto-block и TTD (время от сигнала до алерта) должны иметь запас относительно 30/60-дневных окон.

## 11.4. 6AMLD

Шестая антиотмывочная директива ЕС гармонизирует 22 predicate offences (включая cybercrime, environmental crime, tax crime), криминализует aiding/abetting/inciting/attempting, вводит ответственность юридических лиц и минимальный срок 4 года, требует dual criminality. Для Spillety это два конкретных места в системе: DAG учитывает тип predicate offence при приоритизации алертов, а метаданные якорей включают тип offence из судебных документов; в признаках GBDT присутствует `predicate_offence_severity`. Dual criminality усложняет кросс-юрисдикционную приоритизацию — об этом в ограничениях.

## 11.5. Daubert: допустимость в суде

### 11.5.1. Критерии

| Критерий | Проверка |
|----------|----------|
| Testability | Верифицируемость методологии |
| Error rate | Известная частота ошибок |
| Peer review | Внешняя проверка |
| General acceptance | Принятие в relevant field |

Фокус судебного inquiry — принципы и методология, а не конкретный вывод.

### 11.5.2. Соответствие Spillety

| Критерий | Реализация |
|----------|------------|
| Testability | PR-AUC, Brier, ECE на held-out temporal split |
| Error rate | Precision@K, Recall@K, FP-rate per analyst, bootstrap CI |
| Peer review | Independent validation внешней командой |
| General acceptance | Contrastive learning + GBDT — устоявшиеся методы |

Существенная честная оговорка: без внешней Daubert-валидации заявление о court-admissibility недопустимо. Требуется adversarial testing (red team ежемесячно) на устойчивость к adversarial anchors — попыткам специально сконструировать ложную близость в embedding-пространстве.

## 11.6. Bias-аудит

### 11.6.1. Постановка

Регуляторы (EBA, AI Act) требуют детекции и митигации нежелательного смещения. В KYT-домене конкретные риски: повышенный FP-rate для отдельных юрисдикций, country-contingent differential treatment, и — коварная деталь — levelling down: попытка выровнять precision между группами может просто ухудшить качество для всех.

### 11.6.2. Метрики справедливости

| Метрика | Определение |
|---------|-------------|
| Demographic parity | Равенство positive rate across groups |
| Equalized odds | Равенство TPR и FPR across groups |
| Predictive parity | Равенство precision (PPV) across groups |

Метрики несовместимы одновременно — это математический факт, а не техническая трудность, поэтому выбор критерия — содержательное решение. Для юрисдикционного аудита Spillety принят **equalized odds**: регулятору критичны и пропуски, и ложные срабатывания, то есть обе компоненты — TPR и FPR. Оценка — bootstrap CI по группам, проверка значимости, отчёт по FP-rate и alert-to-SAR.

### 11.6.3. Jurisdiction-level аудит

Процедура: стратификация алертов по юрисдикциям; оценка precision, recall, FPR по группам; проверка equalized odds (близость TPR и FPR) с bootstrap CI; при значимой диспропорции — feature-level разбор. Ключевое различение: диспропорция может быть **justified** (концентрация OFAC-санкций в определённых юрисдикциях) или **unjustified**; различить их требует экспертизы и документации, автоматического критерия нет.

### 11.6.4. SHAP для скрытого смещения

Признак, коррелирующий с юрисдикцией (например, `news_co_mention_count`), может работать прокси для sensitive attribute. SHAP-анализ (раздел 7) выявляет такие признаки и направляет митигацию: регуляризация, исключение, перевзвешивание.

## 11.7. Ограничения

1. SR 26-2 — guidance, не enforceable standard; non-compliance сам по себе не влечёт criticism, но unsafe practices влекут action.
2. Оценка materiality субъективна; критерии exposure/purpose определяет институт.
3. GenAI вне scope SR 26-2, но принципы governance применимы к LLM Explainer по аналогии.
4. FATF Rec. 16 имплементирован неравномерно; Sunrise Issue — постоянный операционный риск.
5. Дедлайны FinCEN SAR жёсткие; auto-block и TTD должны давать запас.
6. 6AMLD dual criminality усложняет кросс-юрисдикционную приоритизацию.
7. Daubert требует adversarial testing; без него допустимость под вопросом.
8. Различение justified vs unjustified disparity не формализовано; метрики справедливости несовместимы одновременно.

---

## См. также

- [12_metrics_dashboard.ipynb](../notebooks/12_metrics_dashboard.ipynb) — метрики качества и мониторинг compliance.