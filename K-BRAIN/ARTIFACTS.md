# ARTIFACTS

## Именование
`<phase>-<type>-<slug>.<ext>`

| Префикс | Фаза | Тип |
|---------|------|-----|
| `01-` | Questions | `.md` |
| `02-` | Research | `.mmd`, `.plot.py` |
| `03-` | Design | `.md`, `.html`, `.mmd` |
| `04-` | Structure | `.md` |
| `05-` | Plan | `.md` |
| `06-` | Implementation | `.md` |

## Правила
- Не редактируются задним числом после перехода в I (новые итерации — `-v2`).
- Ссылки на артефакты обязательны в `05-plan.md`.
- Интерактив — `.html`, статика — `.mmd` (mermaid).
- Графики — seaborn: скрипт `*.plot.py` рядом с картинкой.
- Subagent-вывод — `artifacts/subagents/<agent>-<slug>.md`.
