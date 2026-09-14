# TODO — TCH-GT по temp.md (v2, от 2026-09-14)

> Эталон: `temp.md`. База гэпов: `REPORT-gap.md` (8 приоритетных доборов, 9 DoD-метрик не подтверждены).
> ⚠️ Сознательное отклонение от `K-BRAIN/ARTIFACTS.md` (не переписывать задним числом): по явному решению человека от 2026-09-14 вся Done-история v1 удалена **без архива**. Трассируемость решений v1 утрачена (ссылки `05-plan-master.md → TASKS/*` могут вести на контекст, которого больше нет в TODO).
> Канон (решения человека 2026-09-14): онтология графа — **по temp.md** (Wallet/Person/Mixer/Exchange/NewsArticle/Pattern; TRANSACTS/SAME_AS/MENTIONED_IN/MATCHES_PATTERN); геометрия — **Poincaré-ball** (Lorentz — только как внутренний шаг при необходимости, ablation обязателен).
> ❓ Открыто (решено 2026-09-14): feature store — **УДАЛИТЬ** по философии temp.md. Выполнено: удалены `model_core/feature_store/` + `model_core/tests/test_feature_store.py` + секция `feature_store` из `model_config.yaml`. Не тронут staged-дубликат `model_core/src/feature_store/` (чужая staged-область, ждёт решения по U2).

## Волны (утверждено: A→E, не всё сразу)

- **Волна A:** G1-контракты + G3-streaming (контракты первые: JSON-схема temp.md и сигнатуры orchestrator).
- **Волна B:** G2-ядра (causal, Poincaré, Hawkes, PageRank).
- **Волна C:** G4 (ER) + G5 (онтология temp.md).
- **Волна D:** G6 (JSON+regulatory+Iceberg) + G7 (ClickHouse/Alert/Case/SAR/LLM).
- **Волна E:** G8 (замеры всех DoD).

## G1. Сквозной orchestrator и контракты (волна A) — ✅ A1 done 2026-09-14

- [x] G1.1. JSON-схема алерта буквально по temp.md: `alert_id/timestamp/transaction{tx_hash,from,to,amount,asset}/risk_score 0–1/decision/explanation{counterfactual{removed_edge,score_without_edge,delta,interpretation}/causal_path[{edge,causal_effect}]/hyperbolic_distance{nearest_scam_cluster,distance,percentile}/hawkes_intensity{lambda_t,threshold,trigger}/regulatory_references[]}` + валидатор. Текущее: `model_core/explainability/schema/aml_schema.json` — другая схема, `additionalProperties:false` режет эталонные поля.
- [x] G1.2. `SubgraphExtractor`: union from+to, depth 2 (сейчас только 1 адрес, `bfs_subgraph.py:19`); убрать `{}`-паддинг (`micro_batch.py:29`); чанкинг вместо truncation (`gpu_forward.py:19`).
- [x] G1.3. Скелет orchestrator: `extract → cache → TCH-GT forward (temporal+causal+hyperbolic) → score → explain → JSON`. Сейчас сборки нет, `gpu_forward.py` — generic Linear.
- [ ] G1.4. Заготовка latency-harness (бюджет 20+60+20=100ms, цель p99<100ms depth 2).

## G2. Ядра TCH-GT (волна B) — ✅ done 2026-09-14 (B1–B4, замеры — волна E)

- [x] G2.1. Causal: настоящий backdoor/`do()` + `causal_effect` на ребро; `get_edge_mask` реально используется в `forward` (сейчас аддитивная поправка, `causal_attention.py:48`).
- [x] G2.2. Poincaré-ball ops для message passing + ablation Euclidean vs hyperbolic (DoD: AUC +3%).
- [x] G2.3. Hawkes: связка `μ(t)+kernel → λ(t)`; exponential kernel рядом с power-law; триггер пересчёта (`burst_detected`); бенчмарк обновления λ<1ms. Плюс фикс `temporal_split` без shuffle внутри сплитов (сейчас утечка, `split.py:18-19`).
- [x] G2.4. Incremental PageRank по топологии (random walk затронутых вершин; сейчас `1/n` и пустой `addEdgeToTopology`); замер speedup (DoD ≥10x).

## G3. Streaming Kafka+Flink (волна A) — ✅ A2 done 2026-09-14 (кроме G3.4)

- [x] G3.1. Ingest news/KYC/AVM: proto + топики + консьюмеры (сейчас только on-chain `raw-events`).
- [x] G3.2. Event-time: watermarks/allowedLateness; убрать `currentTimeMillis` подмену времени; глобальный dedup; enrichment-оператор.
- [x] G3.3. Билдер temporal edges (`valid_from/valid_to`); рабочие sink'и Flink→Memgraph/ClickHouse (сейчас no-op); починка Protobuf/JSON-разрыва и типового mismatch `FlinkJob`; `KafkaSource` вместо legacy `FlinkKafkaConsumer`.
- [x] G3.4. Hawkes O(1) updater + обучение α/β/μ (стык с G2.3).

## G4. Entity resolution (волна C) — ✅ done 2026-09-14

- [x] G4.1. Эвристики timing + fee patterns (сейчас только co-spending).
- [x] G4.2. Реальные эмбеддинги вместо `np.random.rand(128)` (`ml_pipeline.py`); fusion heuristic+ML в `pipeline.py` (веса `pipeline.yaml` сейчас игнорируются).
- [x] G4.3. Проверка precision ≥0.95 / recall ≥0.80 (`thresholds.yaml`).

## G5. Онтология Memgraph по temp.md (волна C, rewrite in place) — ✅ done 2026-09-14

- [x] G5.1. Ноды Wallet/Person/Mixer/Exchange/NewsArticle/Pattern; рёбра TRANSACTS/SAME_AS/MENTIONED_IN/MATCHES_PATTERN (миграция с Address/Entity/Risk + TRANSFER/CO_SPEND/SANCTIONS_FLAG).
- [x] G5.2. Рабочий MENTIONED_IN writer/query + news-инжест (стык с G3.1); temporal на всех рёбрах (сейчас только TRANSFER); Flink-sink (стык с G3.3).

## G6. JSON + regulatory + Iceberg (волна D) — ✅ done 2026-09-14

- [x] G6.1. Маппинг FATF Rec.16 / AMLD6 Art.3(1) (сейчас только FinCEN/AMLD5); IVMS101 `message_builder/schema/parser.py` + валидация.
- [x] G6.2. Iceberg append-only writer + catalog + `clickhouse_to_iceberg.py` (90d→S3); каждый алерт immutable.

## G7. OLAP + Alert + Case + SAR + LLM (волна D) — ✅ done 2026-09-14 (замеры — волна E)

- [x] G7.1. ClickHouse: `transactions_raw` (ReplacingMergeTree), `velocity_1h/24h/7d` (AggregatingMergeTree), `pattern_matching.sql`, окна `ROWS BETWEEN` (сейчас нет ни одного *.sql).
- [x] G7.2. Alert layer: score/dedup/merge + топики `alerts_raw/merged/p0`.
- [x] G7.3. Case orchestration + ServiceNow-клиент + `alert_json_builder`.
- [x] G7.4. SAR/STR: `*.j2` (FinCEN/EU/UK) + `fincen_xsd.py` + `field_mapper.py` + human-review.
- [x] G7.5. Настоящий LLM-клиент (LLaMA, читает JSON и объясняет, **не решает**; сейчас стаб-строка).

## G8. Замеры DoD (волна E) — ✅ done 2026-09-14 (harness + синтетика = done по решению человека)

> Свидетельства (все в K-BRAIN, в коммит не идут): `K-BRAIN/TASKS/tchgt-v2/bench/hawkes-BENCH.md`, `flink-BENCH.md`, `inference-BENCH.md`, `METRICS.md`. Честные вердикты: Hawkes — условный PASS (p99 плавает); PageRank — условный PASS (цепочки с запасом); lazy p99 — PASS (со стабом forward); spurious 99.44% PASS; ablation +0.00% FAIL на синтетике (нужен Elliptic++); accuracy-протокол готов (0.7431 на заглушке); JSON/IVMS/Iceberg 100% PASS. Плюс фикс `label_map.json`.

| DoD temp.md | Статус сейчас | Цель волны E |
|---|---|---|
| Accuracy ≥0.95 Elliptic++ (temporal split) | не подтверждается | обучение + замер |
| Spurious reduction ≥20% | не подтверждается | замер vs baseline |
| AUC +3% hyperbolic | не подтверждается | ablation-замер |
| Hawkes λ<1ms | не подтверждается | бенчмарк |
| PageRank speedup ≥10x | не подтверждается | замер |
| Lazy p99<100ms depth 2 | не подтверждается | нагрузочный замер |
| JSON 100% валидны | частично (своя схема) | 100% к схеме temp.md |
| Iceberg immutable | нет кода | проверка неизменяемости |
| IVMS101 | нет кода | проверка совместимости |

## Правила волн

- Контракты (G1.1, сигнатуры orchestrator) — первые; остальные волны строятся поверх них.
- Дифф >200 строк → под-шаги с отдельным ревью; после каждого под-шага — code-review + ai-slops.
- Комментарии в коде — на английском, только сложная логика; всё остальное — на русском.

---

## Slop-чистка (2026-09-14) — ✅ done, 4 subagents

- **S1 model_core:** удалены все fallback (template-fallback LLM, FSDP-CPU, v1-валидатор, `or`-заглушки, молчаливые except), ~35 `demo()`, `HardNegativeSampler`, ~15 verbose ponytail/docstring-блоков. Strict: KeyError/ValueError вместо дефолтов. `TchgtForward` сохранён (контракт).
- **S2 flink:** no-op ClickHouse-sink → throws; 3× silent catch → RuntimeException; timestamp-"0" → IllegalArgumentException; удалён мёртвый код (computeIntensity, harness в прод-классе, дубли TTL/конфигов, cep/json из pom).
- **S3 периферия:** удалён parquet-фолбэк Iceberg (инжекция таблицы), in-memory outbox, DLQ-фолбэки, все `demo()`; ruff скоупа — 0 ошибок (было 34–36).
- **S4 доки:** ai-отчёты только в K-BRAIN (`TASKS/tchgt-v2/` + `bench/`); в трекнутом дереве ai-отчётов нет.
- Проверки: model_core 113 passed + 3 предсуществующих; периферия 30 passed; javac surrogate 0 ошибок.
- Пост-фикс основной сессии: настоящий баг B1 `causal_attention.py:81` (`_, codes.append(...)` — кортеж-выражение + NameError в discrete-ветке, тесты её не покрывали) — исправлено на `_, inv = ...; codes.append(inv)`, discrete-путь проверен вручную.
