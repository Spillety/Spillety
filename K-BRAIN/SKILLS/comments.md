---
skill: comments
description: Правила комментариев в коде и дизайн-доках
---

# В коде
Пиши только для **сложной** логики: нетривиальный алгоритм, контракт, edge cases.

**Нельзя:** перевод имени класса, пересказ кода словами, нумерация внутри метода (`# 1. ...`), module-docstring, пересказывающий архитектуру.

# В дизайн-доках
```md
### [human] <имя>
> Вопрос / замечание / контр-предложение.

### [agent] resolved
> Что изменено в дизайне.
```

- Статус: `open` / `resolved`.
- Каждый resolved → правка дизайна ИЛИ явное «не делаем» с обоснованием.
- Переход D→S — только при 0 открытых комментариев.
- Решения из комментариев — в `05-plan.md`.

Все комментарии в таком формате: 
```py
def get_node_attention(
    scorer: UnifiedScorer,
    agent_ids: List[str],
    X_tabular: np.ndarray,
    X_text: np.ndarray,
    X_graph: np.ndarray | None = None,
) -> Dict[str, float]:
    """
    ## Extract per-agent attention scores for visualization

    Parameters
    ----------
    scorer : UnifiedScorer
        Trained UnifiedScorer instance.
    agent_ids : List[str]
        List of agent identifiers corresponding to rows in X_*.
    X_tabular, X_text, X_graph : np.ndarray
        Feature arrays used for scoring.

    Returns
    ----------
    Dict[str, float]
        Mapping agent_id -> mean attention score across heads.
    """
```