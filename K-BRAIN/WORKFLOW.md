# WORKFLOW — QRSPI

## Фазы

| Фаза | Вход | Выход | Стоп-условие |
|------|------|-------|--------------|
| **Q** — Questions | сырое намерение | `01-questions.md` | ответы на все вопросы или «дефолты норм» |
| **R** — Research | ответы на Q | `02-research.md` + артефакты | названы все файлы и зависимости |
| **D** — Design | research | `03-design.md` | 0 открытых комментариев |
| **S** — Structure | дизайн | `04-structure.md` | каждая фаза ≤200 строк, проверяема |
| **P** — Plan | structure | `05-plan.md` | пользователь подтвердил |
| **I** — Implement | plan | `06-implementation.md` + код | 0 🔴 Critical, 0 ai-slops, AC выполнены |

## Правила
- На каждой фазе — минимум 2 гипотезы что может пойти не так (pushback).
- Дифф >200 строк → разбить на под-шаги с отдельным ревью.
- Отклонение от плана → стоп и вопрос.
- Мышление — за человеком. Агент предлагает, человек решает.

## Артефакты задачи

```
TASKS/<task-slug>/
├── 01-questions.md
├── 02-research.md
├── 03-design.md
├── 04-structure.md
├── 05-plan.md
├── 06-implementation.md
└── artifacts/
    ├── *.mmd, *.html, *.plot.py
    └── subagents/<agent>-<slug>.md
```
