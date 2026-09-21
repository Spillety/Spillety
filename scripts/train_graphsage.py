#!/usr/bin/env python3
"""## GraphSAGE training with NT-Xent + anchor-hinge on Elliptic++."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected

from spillety.data.loader import load_elliptic
from spillety.embeddings.pairs import (
    positive_pairs,
    sample_negatives,
    hard_negatives,
    knn_hard_negatives,
    sampling_probs,
)
from spillety.embeddings.loss import nt_xent, anchor_loss


SEED = 72
NEIGHBOR_SAMPLES = (25, 10)


def _sample_layer_edges(
    torch_mod: Any, edge_index: torch.Tensor, n_nodes: int, fanout: int, gen: torch.Generator
) -> torch.Tensor:
    dst = edge_index[1]
    order = torch.randperm(edge_index.shape[1], generator=gen)
    counts = torch.zeros(n_nodes, dtype=torch.long)
    keep = torch.zeros(edge_index.shape[1], dtype=torch.bool)
    for e in order.tolist():
        d = int(dst[e])
        if counts[d] < fanout:
            keep[e] = True
            counts[d] += 1
    return edge_index[:, keep]


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim, aggr="mean")
        self.conv2 = SAGEConv(hidden_dim, out_dim, aggr="mean")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, e1))
        return self.conv2(h, e2)


def build_pairs(n_nodes, edges, labels, times, degrees, use_curriculum, seed):
    pos = positive_pairs(edges, labels, times, eps=1)
    print(f"Positive pairs (co-spend, same label, Δt≤1): {len(pos)}")

    probs = sampling_probs(degrees, alpha=0.75)
    n_neg = min(len(pos) * 4, n_nodes * 10)
    neg = sample_negatives(n_nodes, degrees, n_neg, alpha=0.75, rng=seed)
    print(f"Negative pairs (degree-corrected): {len(neg)}")

    hard = hard_negatives(labels, times, n=min(len(pos), 5000), eps=1, use_curriculum=use_curriculum, rng=seed)
    print(f"Hard negatives (same window, diff cluster): {len(hard)}")

    if use_curriculum and len(hard) > 0:
        neg = np.vstack([neg, hard])
        print(f"Total negatives with hard: {len(neg)}")

    return pos, neg


def train_graphsage(
    X: np.ndarray,
    edge_index: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    hidden_dim: int = 128,
    out_dim: int = 128,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    tau: float = 0.1,
    lam: float = 0.5,
    m0: float = 1.0,
    m_push: float = 1.0,
    use_curriculum: bool = False,
    seed: int = 72,
    device: str = "cpu",
) -> tuple[GraphSAGE, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_nodes = X.shape[0]
    x = torch.as_tensor(X, dtype=torch.float32, device=device)
    ei = to_undirected(torch.as_tensor(edge_index, dtype=torch.long, device=device))
    degrees = torch.bincount(ei[1], minlength=n_nodes).float().cpu().numpy()
    degrees = np.maximum(degrees, 1.0)

    pos, neg = build_pairs(n_nodes, edge_index, labels, times, degrees, use_curriculum, seed)

    pos_tensor = torch.as_tensor(pos, dtype=torch.long, device=device)
    neg_tensor = torch.as_tensor(neg, dtype=torch.long, device=device)

    gen = torch.Generator(device=device).manual_seed(seed)
    e1 = _sample_layer_edges(torch, ei, n_nodes, NEIGHBOR_SAMPLES[0], gen)
    e2 = _sample_layer_edges(torch, ei, n_nodes, NEIGHBOR_SAMPLES[1], gen)

    model = GraphSAGE(X.shape[1], hidden_dim, out_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    labels_t = torch.as_tensor(labels, dtype=torch.long, device=device)
    times_t = torch.as_tensor(times, dtype=torch.long, device=device)

    best_loss = float("inf")
    best_state = None

    print(f"Training GraphSAGE: {X.shape[1]} -> {hidden_dim} -> {out_dim}, epochs={epochs}, device={device}")
    print(f"  NT-Xent tau={tau}, anchor_loss lam={lam}, m0={m0}, m_push={m_push}")
    print(f"  Pos pairs: {len(pos)}, Neg pairs: {len(neg)}, Curriculum: {use_curriculum}")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        e1 = _sample_layer_edges(torch, ei, n_nodes, NEIGHBOR_SAMPLES[0], gen)
        e2 = _sample_layer_edges(torch, ei, n_nodes, NEIGHBOR_SAMPLES[1], gen)

        z = model(x, ei, e1, e2)

        z_pos_a = z[pos_tensor[:, 0]]
        z_pos_b = z[pos_tensor[:, 1]]
        z_neg_a = z[neg_tensor[:, 0]]
        z_neg_b = z[neg_tensor[:, 1]]

        z_pairs = torch.cat([z_pos_a, z_pos_b, z_neg_a, z_neg_b], dim=0)
        z_pairs_np = z_pairs.detach().cpu().numpy()

        loss_ntx = nt_xent(z_pairs_np, tau=tau)

        sanctions_mask = labels > 0
        anchor_idx = np.where(sanctions_mask)[0]
        non_anchor_idx = np.where(~sanctions_mask)[0]

        if len(anchor_idx) > 0 and len(non_anchor_idx) > 0:
            za = z[anchor_idx].detach().cpu().numpy()
            zn = z[non_anchor_idx].detach().cpu().numpy()
            loss_anchor = anchor_loss(za, zn, lam=lam, m0=m0, jaccard=0.0, m_push=m_push)
        else:
            loss_anchor = 0.0

        loss = loss_ntx + loss_anchor
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: NT-Xent={loss_ntx:.4f}, Anchor={loss_anchor:.4f}, Total={loss:.4f}")

        if loss < best_loss:
            best_loss = loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        e1 = _sample_layer_edges(torch, ei, n_nodes, NEIGHBOR_SAMPLES[0], gen)
        e2 = _sample_layer_edges(torch, ei, n_nodes, NEIGHBOR_SAMPLES[1], gen)
        z_final = model(x, ei, e1, e2).cpu().numpy()

    print(f"Training complete. Best loss: {best_loss:.4f}")
    return model, z_final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/elliptic_raw")
    ap.add_argument("--out", default="models/elliptic_graphsage")
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--out-dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--m0", type=float, default=1.0)
    ap.add_argument("--m-push", type=float, default=1.0)
    ap.add_argument("--curriculum", action="store_true", help="Enable hard negative curriculum")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    opt = ap.parse_args()

    out = Path(opt.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"Loading Elliptic++ from {opt.root}...")
    _f, _c, edgelist, merged = load_elliptic(opt.root)

    labeled = merged[merged["class"].isin(["1", "2"])].copy()
    LABEL = {"1": 1, "2": 0}
    labels = labeled["class"].map(LABEL).astype(int).to_numpy()
    times = labeled["time_step"].to_numpy()
    raw_cols = [c for c in merged.columns if c.startswith("feat_")]
    X = labeled[raw_cols].fillna(0.0).to_numpy(dtype=np.float32)

    txid_to_idx = {txid: i for i, txid in enumerate(labeled["txId"])}
    edges = []
    for _, row in edgelist.iterrows():
        if row["txId1"] in txid_to_idx and row["txId2"] in txid_to_idx:
            edges.append([txid_to_idx[row["txId1"]], txid_to_idx[row["txId2"]]])
    edge_index = np.array(edges, dtype=int).T
    print(f"Graph: {len(labeled)} nodes, {edge_index.shape[1]} edges")

    model, Z = train_graphsage(
        X=X,
        edge_index=edge_index,
        labels=labels,
        times=times,
        hidden_dim=opt.hidden_dim,
        out_dim=opt.out_dim,
        epochs=opt.epochs,
        lr=opt.lr,
        weight_decay=opt.weight_decay,
        tau=opt.tau,
        lam=opt.lam,
        m0=opt.m0,
        m_push=opt.m_push,
        use_curriculum=opt.curriculum,
        seed=SEED,
        device=opt.device,
    )

    torch.save(model.state_dict(), out / "graphsage.pt")
    np.save(out / "embeddings.npy", Z)

    meta = {
        "in_dim": X.shape[1],
        "hidden_dim": opt.hidden_dim,
        "out_dim": opt.out_dim,
        "neighbor_samples": NEIGHBOR_SAMPLES,
        "epochs": opt.epochs,
        "lr": opt.lr,
        "tau": opt.tau,
        "lam": opt.lam,
        "m0": opt.m0,
        "m_push": opt.m_push,
        "curriculum": opt.curriculum,
        "seed": SEED,
        "n_nodes": len(labeled),
        "n_edges": edge_index.shape[1],
        "train_time_sec": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved model + embeddings to {out}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()