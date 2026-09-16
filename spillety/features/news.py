import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict


def build_comention_graph(mentions, entity_col="entity", article_col="article_id", timestamp_col=None):
    """
    Build a news co-mention graph from article-entity mentions.

    Parameters
    ----------
    mentions : DataFrame
        Rows of (article_id, entity, [timestamp]).
    entity_col : str
        Column with entity identifier (e.g., wallet address).
    article_col : str
        Column grouping entities that appear together.
    timestamp_col : str or None
        Optional timestamp for temporal weighting.

    Returns
    -------
    G : networkx.Graph
        Undirected weighted graph where edge weight = co-mention count.
    cooccurrence : DataFrame
        Columns [entity_a, entity_b, weight, n_articles].
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
    # add isolated nodes
    for e in mentions[entity_col].unique():
        if e not in G:
            G.add_node(e)
    return G, cooccurrence


def entity_news_features(mentions, entity, entity_col="entity", article_col="article_id"):
    """
    Compute per-entity news features: mention_count, unique_articles, avg_comentions.
    """
    sub = mentions[mentions[entity_col] == entity]
    mention_count = len(sub)
    unique_articles = sub[article_col].nunique() if mention_count else 0
    # avg co-mentions per article
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


def news_proximity(entity_a, entity_b, G, default=1.0):
    """
    Network proximity in news co-mention graph.
    Returns 1 / (1 + shortest_path_length) or default if disconnected.
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


if __name__ == "__main__":
    # synthetic smoke
    rng = np.random.default_rng(72)
    n_articles = 50
    articles = []
    wallets = [f"0x{i:04d}" for i in range(20)]
    for aid in range(n_articles):
        n_ents = rng.integers(2, 6)
        ents = rng.choice(wallets, size=n_ents, replace=False)
        for e in ents:
            articles.append({"article_id": f"A{aid:03d}", "entity": e})
    df = pd.DataFrame(articles)
    G, cooc = build_comention_graph(df)
    print(f"news graph: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    assert G.number_of_nodes() <= len(wallets)
    assert cooc["weight"].min() >= 1

    feat = entity_news_features(df, wallets[0])
    print(f"entity_news_features {wallets[0]}: {feat}")
    assert feat["mention_count"] >= 0

    prox = news_proximity(wallets[0], wallets[1], G)
    print(f"news_proximity {wallets[0]} <-> {wallets[1]} = {prox:.4f}")
    assert 0 <= prox <= 1

    # empty smoke
    G_empty, cooc_empty = build_comention_graph(pd.DataFrame(columns=["article_id", "entity"]))
    assert G_empty.number_of_nodes() == 0
    assert cooc_empty.empty
    print("news smoke passed")
