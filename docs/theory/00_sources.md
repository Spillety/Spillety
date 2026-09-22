# Источники

---

## Блокчейн-основы и данные

#### Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics
[ссылка](https://arxiv.org/abs/1908.02591)

Исследование представляет датасет Elliptic — крупнейший публичный размеченный граф Bitcoin-транзакций (200K+ узлов, 234K рёбер, 166 признаков) и сравнивает LR, RF, MLP и GCN в задаче бинарной классификации illicit-транзакций.

Результаты показали превосходство Random Forest над GCN и другими методами; авторы прямо пишут: «The results show the superiority of Random Forest (RF), but also invite algorithmic work to combine the respective powers of RF and graph methods». RF на всех 166 признаках даёт illicit-F1 = 0.788, а с добавлением GCN-эмбеддингов — 0.796, что остаётся лучшим результатом в работе.

Мы используем этот датасет как основной источник для калибровки порогов и измерения метрик (PR-AUC, ECE, Brier), а также как отправную точку для сравнения с конкурентами — с обязательной оговоркой о различии протоколов оценки.


#### GuiltyWalker: Distance to illicit nodes in the Bitcoin network
[ссылка](https://arxiv.org/abs/2102.05373)

Работа исследует, как расстояния до известных illicit-узлов в графе Bitcoin могут служить признаками для улучшения детекции отмывания.

Ключевой результат: добавление GuiltyWalker-признаков к RF-модели улучшает F1 на 10–16 п.п. после шага 43 (момент закрытия тёмного рынка), где baseline RF падает ниже 0.25. В зоне низких false positive rates (1–10%) recall растёт на 5–6 п.п.

Это прямо обосновывает наш подход к пространству с якорями: вместо ручных признаков мы используем расстояние до санкционных адресов как сигнал, и именно на сдвигах распределения (аналог «шага 43») этот сигнал даёт наибольший прирост.


#### Illicit Bitcoin transaction detection via feature-gated temporal graph learning
[ссылка](https://www.nature.com/articles/s41598-026-53783-y)

Предлагается FG-EGCN — темпоральная графовая модель с residual feature branch и adaptive gating, которая балансирует структурные и признаковые сигналы.

Эксперименты на Elliptic с хронологическим сплитом показывают более высокую temporal stability по сравнению с чисто графовыми моделями; ablation подтверждает вклад feature preservation и gating.

Для нашего проекта это подтверждает разделение retrieval и GBDT-решения: gating-механизм — аналог того, как мы комбинируем эмбеддинги (структура) и сырые признаки (feature branch) в финальном классификаторе.

---

## Графовые эмбеддинги и контрастивное обучение

#### Inspection-L: Self-Supervised GNN Node Embeddings for Money Laundering Detection in Bitcoin
[ссылка](https://arxiv.org/abs/2203.10465)

Исследование использует DGI (Deep Graph Infomax) с GIN-энкодером для self-supervised обучения эмбеддингов узлов, после чего Random Forest классифицирует на объединении сырых признаков и эмбеддингов.

Два варианта: AF + DNE (все признаки + эмбеддинги) даёт F1 = 0.828, Recall = 0.721; LF + DNE (локальные признаки + эмбеддинги) — F1 = 0.712, Recall = 0.797. Важно: RF на эмбеддингах без сырых признаков показывает очень низкий false alarm rate, что критично для операционной нагрузки.

Это прямой прецедент для нашей архитектуры: GNN-эмбеддинги как вход для tree-based классификатора, а не как end-to-end решение. Мы идём дальше — заменяем self-supervised DGI на контрастивное обучение с санкционными якорями как positive samples.


#### PageRank-Based Unsupervised Deep Vertex Representations for Anti-Money Laundering Detection
[ссылка](https://ieeexplore.ieee.org/document/11251248)

Предлагается AML PD — unsupervised framework на основе Personalized PageRank и diffusion для генерации индуктивных эмбеддингов с учётом направленности рёбер и edge features.

На proprietary banking dataset (~10M узлов, ~23M рёбер) и публичных бенчмарках AML PD + XGBoost превосходит supervised graph-based методы; ablation показывает, что positional encodings (особенно Laplacian eigenvectors) дают значительный прирост, а удаление PE или компрессия (PCA) ухудшают качество. Метод индуктивен, что позволяет обобщаться на невиданные подграфы.

Для Spillety это подтверждает индуктивность и направленность как ключевые требования: мы используем HNSW для retrieval по эмбеддингам, и именно индуктивность позволяет нам добавлять новые санкционные адреса без переобучения.


#### When Graph Structure Becomes a Liability: A Critical Re-Evaluation of Graph Neural Networks for Bitcoin Fraud Detection under Temporal Distribution Shift
[ссылка](https://arxiv.org/abs/2604.19514)

Критическая ре-оценка консенсуса о превосходстве GNN на Elliptic. Под строго индуктивным протоколом (без доступа к тестовым узлам при обучении) Random Forest на 165 сырых признаках достигает F1 = 0.821 ± 0.003, обгоняя все протестированные GNN; GraphSAGE — только 0.689 ± 0.017.

Паpaired experiment показывает разрыв в 39.5 F1 пункта, объясняемый training-time exposure к тестовой adjacency (утечка). Edge-shuffle ablation: случайные рёбра работают лучше реального графа, что означает — топология Elliptic вредна при temporal distribution shift. Гибридные модели (GNN-эмбеддинги + сырые признаки) дают лишь маргинальный прирост и остаются ниже feature-only baseline.

Это ключевая работа для позиционирования Spillety: мы не «проигрываем» RF, а сознательно строим систему, которая работает в честном протоколе. Наш подход с retrieval + GBDT — это не попытка «побить RF на Elliptic», а попытка построить воспроизводимую KYT-инфраструктуру, где RF может быть компонентом, но не единственным решением.

---

## Метрики и бенчмарки

#### ADCC-Bench: A Benchmark Framework for Anomaly Detection in Cryptocurrency Transactions
[ссылка](https://ieeexplore.ieee.org/document/11580002)

Унифицированный бенчмарк с тремя методологическими контролями: modality-aware splitting, стандартизованная предобработка, multi-seed replication (N=20) с Welch’s t-tests.

Три ключевых вывода: 
1. random splitting завышает Macro-F1 до 11% и меняет ранжирование моделей — подтверждение серьёзности temporal leakage; 
2. tree-based модели (RF, XGBoost, LightGBM) стабильно дают лучший single-model Macro-F1, а GCN-augmented ensembles дают комплементарный прирост на transaction-flow data (Elliptic++); 
3. прирост от ансамблей статистически незначим — их основная польза в снижении variance.

RF/XGBoost/LightGBM — хороший выбор как single model; ансамбли оправданы для снижения variance, а не для «прорыва в точности».


#### Bridging the Reality Gap: Evaluating AML Detection under Fragmented and Streaming Constraints
[ссылка](https://dl.acm.org/doi/pdf/10.1145/3810987.3815531)

Исследуются ограничения streaming и fragmented данных для AML-детекции. Показывает, что под streaming-условиями RF сохраняет 68% recall, тогда как graph-enhanced модели падают с 98% до 27%.

Прямое инженерное подтверждение: для real-time scoring (T+0), где граф ещё не «устоялся», RF — практичный выбор. Graph-based методы требуют, чтобы транзакции «осели» в графе, что неприменимо для мгновенного скоринга.

Для Spillety это обосновывает двухрежимную архитектуру: RF для мгновенного скоринга новых транзакций и retrieval/GBDT для batch-анализа уже сформированных кластеров.
