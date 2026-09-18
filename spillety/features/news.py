from collections import defaultdict

import networkx as nx
import pandas as pd


def build_comention_graph(mentions, entity_col="entity", article_col="article_id"):
    """
    ## News co-mention graph from article-entity mentions (§7.3.5)

    Parameters
    ----------
    mentions : DataFrame
        Rows of (article_id, entity).
    entity_col : str
        Entity identifier column.
    article_col : str
        Column grouping co-mentioned entities.

    Returns
    -------
    tuple[networkx.Graph, DataFrame]
        Undirected graph (edge weight = co-mention count) and pair table.
    """
    if mentions.empty:
        G = nx.Graph()
        return G, pd.DataFrame(columns=["entity_a", "entity_b", "weight", "n_articles"])
    groups = mentions.groupby(article_col)[entity_col].apply(list)
    pair_counts = defaultdict(int)
    for entities in groups:
        uniq = sorted(set(entities))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                pair_counts[(uniq[i], uniq[j])] += 1
    rows = []
    for (a, b), w in pair_counts.items():
        rows.append({"entity_a": a, "entity_b": b, "weight": int(w), "n_articles": int(w)})
    cooccurrence = pd.DataFrame(rows)
    G = nx.Graph()
    for _, row in cooccurrence.iterrows():
        G.add_edge(row["entity_a"], row["entity_b"], weight=int(row["weight"]))
    for e in mentions[entity_col].unique():
        if e not in G:
            G.add_node(e)
    return G, cooccurrence


def entity_news_features(mentions, entity, entity_col="entity", article_col="article_id"):
    """
    ## Per-entity mention counts for contextual features (§7.3.5)

    Returns
    -------
    dict
        mention_count, unique_articles, avg_comentions.
    """
    sub = mentions[mentions[entity_col] == entity]
    mention_count = len(sub)
    unique_articles = sub[article_col].nunique() if mention_count else 0
    if mention_count == 0:
        return {"mention_count": 0, "unique_articles": 0, "avg_comentions": 0.0}
    article_groups = mentions.groupby(article_col)[entity_col].nunique()
    shared = sub[article_col].map(article_groups)
    avg_comentions = float((shared - 1).mean()) if len(shared) else 0.0
    return {
        "mention_count": int(mention_count),
        "unique_articles": int(unique_articles),
        "avg_comentions": round(avg_comentions, 4),
    }


def news_proximity(entity_a, entity_b, G):
    """
    ## Inverse-path proximity in co-mention graph (§7.3.5)

    Returns
    -------
    float
        1/(1+path); 0 when unreachable or unknown.
    """
    if entity_a not in G or entity_b not in G:
        return 0.0
    if entity_a == entity_b:
        return 1.0
    try:
        pl = nx.shortest_path_length(G, source=entity_a, target=entity_b)
        return 1.0 / (1.0 + pl)
    except nx.NetworkXNoPath:
        return 0.0
