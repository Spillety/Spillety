# Spillety

Spill the tea about your transactions.

Spillety - студенческий Know Your Transaction движок. Основан на открытых данных и не требует больших вычислительных мощностей для работы. В нём сконцентрированы самые сильные решения, применяемые в Chainalysis, Elliptic и TRM Labs. Классический подход был скорректирован академической теорией и идеями из разделов математики и алгоритмики. 

| Раздел | Роль |
| ------ | ---- |
| [`docs/`](./docs/) | Документация | 
| [`docs/theory`](./docs/theory/) | Теория для погружения в проект |
| [`docs/notebooks/`](./docs/notebooks/) | Notebooks с реализацией алгоритмов проекта поотдельности |

> [!important] Jupyter notebooks: подготовка данных
> ```bash
> make data        # распакует archive.zip → data/elliptic_raw/ (или скачает via kaggle)
> make notebooks   # pip install -r docs/notebooks/requirements-notebooks.txt
> jupyter lab docs/notebooks/
> ```
> Ноутбуки читают его только через `docs/notebooks/_elliptic_loader.py` (read-only). Подробности — `data/README.md` и `scripts/download_elliptic.py`.