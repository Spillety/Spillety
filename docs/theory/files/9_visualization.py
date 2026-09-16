import hashlib
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def sha256(data: str) -> str:
    """Вычисляет SHA-256 хеш строки."""
    return hashlib.sha256(data.encode()).hexdigest()


def build_merkle_tree(leaves):
    """
    Строит Merkle tree из списка листьев.

    Args:
        leaves: list of str, хеши листьев

    Returns:
        list of list of str, уровни дерева (от листьев к корню)
    """
    tree = [leaves]
    current_level = leaves

    while len(current_level) > 1:
        next_level = []
        for i in range(0, len(current_level), 2):
            if i + 1 < len(current_level):
                combined = current_level[i] + current_level[i + 1]
            else:
                combined = current_level[i] + current_level[i]
            next_level.append(sha256(combined))
        tree.append(next_level)
        current_level = next_level

    return tree


def get_merkle_proof(tree, leaf_index):
    """
    Возвращает Merkle proof для листа.

    Args:
        tree: list of list of str
        leaf_index: int, индекс листа

    Returns:
        list of str, proof (siblings)
    """
    proof = []
    index = leaf_index

    for level in tree[:-1]:
        if index % 2 == 0:
            sibling_index = index + 1 if index + 1 < len(level) else index
        else:
            sibling_index = index - 1
        proof.append(level[sibling_index])
        index = index // 2

    return proof


def verify_merkle_proof(leaf_hash, proof, root):
    """
    Верифицирует Merkle proof.

    Args:
        leaf_hash: str
        proof: list of str
        root: str

    Returns:
        bool
    """
    current = leaf_hash
    for sibling in proof:
        if current < sibling:
            current = sha256(current + sibling)
        else:
            current = sha256(sibling + current)
    return current == root


def plot_merkle_tree(tree, title="Merkle tree", filename="merkle_tree.png"):
    """
    Визуализация Merkle tree.

    Args:
        tree: list of list of str
        title: str
        filename: str
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    n_levels = len(tree)
    max_nodes = max(len(level) for level in tree)

    # Позиции узлов
    positions = {}
    for level_idx, level in enumerate(tree):
        y = n_levels - 1 - level_idx
        n_nodes = len(level)
        for node_idx, node_hash in enumerate(level):
            x = (node_idx + 0.5) * (max_nodes / n_nodes)
            positions[(level_idx, node_idx)] = (x, y)

    # Рёбра
    for level_idx in range(n_levels - 1):
        for node_idx in range(len(tree[level_idx])):
            parent_idx = node_idx // 2
            x1, y1 = positions[(level_idx, node_idx)]
            x2, y2 = positions[(level_idx + 1, parent_idx)]
            ax.plot([x1, x2], [y1, y2], 'gray', linewidth=1, alpha=0.5)

    # Узлы
    for level_idx, level in enumerate(tree):
        for node_idx, node_hash in enumerate(level):
            x, y = positions[(level_idx, node_idx)]
            color = '#D32F2F' if level_idx == n_levels - 1 else '#1976D2'
            ax.scatter(x, y, s=800, color=color, zorder=5,
                       edgecolors='white', linewidth=2)
            label = node_hash[:8] + '...'
            ax.text(x, y, label, ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold', zorder=6)

    ax.set_xlim(-0.5, max_nodes + 0.5)
    ax.set_ylim(-0.5, n_levels - 0.5)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_audit_trail(audit_events, title="Audit trail", filename="audit_trail.png"):
    """
    Визуализация audit trail.

    Args:
        audit_events: list of dict, {timestamp, action, analyst}
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    actions = [e['action'] for e in audit_events]
    timestamps = [e['timestamp'] for e in audit_events]
    analysts = [e.get('analyst', 'auto') for e in audit_events]

    unique_actions = list(set(actions))
    action_colors = {a: c for a, c in zip(unique_actions,
                     ['#1976D2', '#D32F2F', '#F57C00', '#388E3C', '#7B1FA2', '#00ACC1', '#5E35B1', '#F06292'])}

    for i, (ts, action, analyst) in enumerate(zip(timestamps, actions, analysts)):
        color = action_colors[action]
        ax.scatter(ts, i, s=200, color=color, zorder=5,
                   edgecolors='white', linewidth=2)
        ax.text(ts, i + 0.3, f'{action} ({analyst})', ha='center', fontsize=9)

    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Event index', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# Пример вызова
if __name__ == "__main__":
    # --- Merkle tree ---
    leaves = [sha256(f"alert_{i}") for i in range(8)]
    tree = build_merkle_tree(leaves)

    plot_merkle_tree(tree, title="Merkle tree for 8 alerts",
                     filename="9-5-1_merkle_tree.png")

    # Merkle proof
    leaf_index = 3
    proof = get_merkle_proof(tree, leaf_index)
    root = tree[-1][0]

    print(f"Leaf: {leaves[leaf_index][:16]}...")
    print(f"Proof: {[p[:16] + '...' for p in proof]}")
    print(f"Root: {root[:16]}...")
    print(f"Verification: {verify_merkle_proof(leaves[leaf_index], proof, root)}")

    # --- Audit trail ---
    base_time = datetime.datetime(2026, 9, 14, 12, 0, 0)
    audit_events = [
        {'timestamp': base_time, 'action': 'created', 'analyst': 'auto'},
        {'timestamp': base_time + datetime.timedelta(minutes=5), 'action': 'viewed', 'analyst': 'analyst_1'},
        {'timestamp': base_time + datetime.timedelta(hours=1), 'action': 'reviewed', 'analyst': 'analyst_1'},
        {'timestamp': base_time + datetime.timedelta(hours=2), 'action': 'escalated', 'analyst': 'analyst_1'},
        {'timestamp': base_time + datetime.timedelta(hours=3), 'action': 'sar_filed', 'analyst': 'compliance_1'},
        {'timestamp': base_time + datetime.timedelta(days=1), 'action': 'verified', 'analyst': 'validator_1'},
    ]

    plot_audit_trail(audit_events, filename="9-5-2_audit_trail.png")