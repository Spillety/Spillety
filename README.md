# Spillety

_Spill the tea about your transactions._

KYT-пайплайн для мониторинга криптовалютных транзакций. Решает две ключевые задачи: дорогой инференс на GPU и зависимость от ручной разметки. Подход основан на публичных санкционных списках как системе координат, индуктивных графовых представлениях и cost-based принятии решений. 

| Показатель | Общепринятый подход | Spillety |
|---|---|---|
| Разметка | Ручная экспертиза аналитиков | Публичные санкционные списки как система координат | 
| Представления | Ручные признаки или GNN с дообучением на новых данных | Индуктивный графовый энкодер и контрастивное обучение |
| Поиск | Эвристики и перебор с последующей фильтрацией | Поиск близких anchors в пространстве представлений | 
| Решение | Сквозные модели от признаков к решению | Разделение на retrieval и GBDT-решение | 
| Порог | Порог по распределению скоров | Порог по стоимости ошибок с учётом нагрузки | 
| Калибровка | Скор используется напрямую | Выбор метода калибровки на отложенной выборке | 
| Эволюция | Переобучение по расписанию | Проверка обобщения на новых санкциях и детекция сдвига |

_Столбец "Общепринятый подход" является примерным, так как топовые решения являются проприетарными._

# Навигация по проекту

| Раздел | Роль |
| ------ | ---- |
| [`.agent/skills/spillety/SKILL.md`](https://github.com/Spillety/Spillety/blob/main/.agents/skills/spillety/SKILL.md) | SKILL файл для ознакомления с проектом |
| [`docs/`](./docs/) | Документация | 
| [`docs/theory`](./docs/theory/) | Теория для погружения в проект |
| [`docs/notebooks/`](./docs/notebooks/) | Notebooks с реализацией алгоритмов проекта по отдельности |

# Jupyter notebooks 

Ноутбуки пронумерованы 01, 02 ... 13. Рекомендуем идти именно в этом порядке.

### Подготовка данных

Notebooks читают elliptic датасет через единый [`docs/notebooks/_elliptic_loader.py`](./docs/notebooks/_elliptic_loader.py). Чтобы скачать датасет, выполните команды ниже:

```bash
make data        # распакует archive.zip в папку data/elliptic_raw/ (или скачает из kaggle)
make notebooks   # pip install -r docs/notebooks/requirements-notebooks.txt
```

# Метрики

### RF Proxy 
`12_metrics_dashboard`

| Метрика | Значение | Base rate | Примечание |
|---|---|---|---|
| PR-AUC | 0.647 | — | validation set |
| Brier | 0.025 | — | validation set |
| ECE | 0.016 | — | validation set |
| Precision@100 | 0.98 | — | top-100 retrieval |

### LightGBM Focal 
`05_gbdt_calibration`

| Метрика | Значение | Base rate | Примечание |
|---|---|---|---|
| F1 illicit | 0.713 | 0.053 | best-F1, τ=0.671 |
| Precision | 0.955 | 0.053 | best-F1 |
| Recall | 0.569 | 0.053 | best-F1 |

### Ensemble v3 
`13_end_to_end`

| Метрика | Значение | Base rate | Примечание |
|---|---|---|---|
| PR-AUC | 0.656 | 0.053 | temporal test split 41–49 |
| ECE | 0.011 | 0.053 | temporal test split 41–49 |
| Precision@100 | 1.0 | 0.053 | test steps 41–49 |

### В сравнении

| Метрика | Weber et al. 2019 (Skip-GCN)  | Inspection-L (RF + DGI)  | Random Forest (raw features)  | Spillety |
|---|---|---|---|---|
| F1 illicit | 0.705 | 0.712 | ~0.71 | 0.681 (best-F1, τ=0.223) / 0.713 (LightGBM Focal, τ=0.671) |
| Precision | — | — | — | 0.872 / 0.955 |
| Recall | — | 0.797 | — | 0.559 / 0.569 |
| ROC-AUC | — | — | — | 0.864 |
| PR-AUC | — | — | — | 0.632 / 0.647 / 0.656 |
| Precision@100 | — | — | — | 0.98 / 1.0 |

*Weber et al. 2019 — оригинальная работа с датасетом Elliptic, Skip-GCN показал F1 = 0.705 . Это академический baseline, на который ссылаются практически все последующие работы.*
*Inspection-L (2022) — self-supervised GNN (DGI) + Random Forest на эмбеддингах, F1 = 0.712, Recall = 0.797 . Результат на том же датасете, но с другим протоколом.*
*Random Forest на raw features — под строгим индуктивным протоколом RF на 165-мерных признаках показывает F1 около 0.71 и остаётся сильнейшим baseline на Elliptic, обгоняя большинство GNN-подходов. Но это сравнение немного некорректно, так как конкретно в этом датасете Elliptic средняя степень вершин 2.3, следовательно, вершины слабо влияют друг на друга.*
*Spillety — две конфигурации: best-F1 на validation (F1 = 0.681, Precision 0.872, Recall 0.559) и LightGBM Focal (F1 = 0.713, Precision 0.955, Recall 0.569). PR-AUC 0.656 на temporal test split 41–49.*
*Base rate — доля illicit-классов в соответствующей оценочной выборке. RF Proxy и LightGBM оценены на validation (шаги 31–40), Ensemble v3 — на test (шаги 41–49).*

![PR curve](./docs/img/pr_curve.png)

![Reliability](./docs/img/reliability.png)

![Walk-forward](./docs/img/walkforward.png)

![Metrics bars](./docs/img/metrics_bars.png)
