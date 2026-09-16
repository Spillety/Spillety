import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np

sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'

np.random.seed(42)
G = nx.Graph()

# Central node
G.add_node("w", type="target")

# 1st-degree neighbors
neighbors = [f"n{i}" for i in range(6)]
for n in neighbors:
    G.add_edge("w", n)
    G.nodes[n]["type"] = "neighbor"

# 2nd-degree neighbors
for i, n in enumerate(neighbors):
    for j in range(np.random.randint(1, 4)):
        second = f"s{i}_{j}"
        G.add_edge(n, second)
        G.nodes[second]["type"] = "second"

pos = nx.spring_layout(G, seed=42, k=0.6)

node_colors = []
for node in G.nodes():
    t = G.nodes[node]["type"]
    if t == "target": node_colors.append("#b85450")
    elif t == "neighbor": node_colors.append("#6c8ebf")
    else: node_colors.append("#82b366")

node_sizes = []
for node in G.nodes():
    t = G.nodes[node]["type"]
    if t == "target": node_sizes.append(1200)
    elif t == "neighbor": node_sizes.append(500)
    else: node_sizes.append(150)

fig, ax = plt.subplots(figsize=(12, 9))
nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)

ax.annotate("Целевой кошелёк", xy=pos["w"], xytext=(pos["w"][0]+0.3, pos["w"][1]+0.3),
            fontsize=12, fontweight='bold', color='#b85450',
            arrowprops=dict(arrowstyle='->', color='#b85450', lw=2))
ax.annotate("Соседи 1-го уровня\n(in/out-degree)", xy=pos["n0"], xytext=(pos["n0"][0]-0.5, pos["n0"][1]+0.3),
            fontsize=10, color='#6c8ebf', arrowprops=dict(arrowstyle='->', color='#6c8ebf', lw=2))
ax.annotate("Соседи 2-го уровня\n(clustering)", xy=pos["s0_0"], xytext=(pos["s0_0"][0]+0.3, pos["s0_0"][1]-0.4),
            fontsize=10, color='#82b366', arrowprops=dict(arrowstyle='->', color='#82b366', lw=2))

ax.set_title("Ego-graph кошелька w (радиус R=2)", fontsize=16, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig("ego_graph.png", dpi=300, bbox_inches='tight')
plt.show()