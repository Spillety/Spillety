import numpy as np
import pytest

SEED = 72

torch = pytest.importorskip("torch")


def _toy_transactions():
    return [["a", "b"], ["b", "c"], ["c", "d"], ["a", "c", "e"]]


def test_account_graph_edge_types_and_attention():
    from spillety.embeddings.account_graph import (
        EDGE_TYPES,
        build_account_graph,
        encode_account_graph,
    )

    txs = _toy_transactions()
    g = build_account_graph(txs)
    assert set(g["edges"]) == set(EDGE_TYPES)
    assert g["edges"]["addr_tx"].shape[0] == 2
    assert g["edges"]["co_spend"].shape[1] > 0
    excluded = build_account_graph(txs, is_coinjoin=[True] * len(txs))
    assert excluded["edges"]["co_spend"].shape == (2, 0)
    assert excluded["edges"]["addr_tx"].shape[1] > 0
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(g["n_addr"], 6)).astype(np.float32)
    z, attn = encode_account_graph(x, g, hidden_dim=8, out_dim=4, seed=SEED)
    assert z.shape == (g["n_addr"], 4)
    assert attn.shape == (g["n_addr"], 2)
    assert torch.allclose(attn.sum(dim=1), torch.ones(g["n_addr"]), atol=1e-5)
    assert bool(((attn > 0) & (attn < 1)).all())


def test_temporal_encoder_shapes_and_determinism():
    from spillety.embeddings.temporal import encode_temporal

    rng = np.random.default_rng(SEED)
    n, d, t = 8, 5, 3
    xs = [rng.normal(size=(n, d)).astype(np.float32) for _ in range(t)]
    edges = [np.array([[i, (i + 1) % n] for i in range(n)]).T for _ in range(t)]
    z1 = encode_temporal(xs, edges, hidden_dim=8, out_dim=4, seed=SEED)
    z2 = encode_temporal(xs, edges, hidden_dim=8, out_dim=4, seed=SEED)
    assert z1.shape == (t, n, 4)
    assert torch.isfinite(z1).all()
    assert torch.equal(z1, z2)
    assert not torch.equal(z1[0], z1[-1])


def test_knn_hard_negatives_from_neighborhood():
    from spillety.embeddings.pairs import knn_hard_negatives

    rng = np.random.default_rng(SEED)
    z = rng.normal(size=(30, 4))
    labels = rng.integers(0, 3, size=30)
    k = 5
    pairs = knn_hard_negatives(z, 50, k=k, labels=labels, rng=SEED)
    assert pairs.shape[1] == 2
    assert len(pairs) > 0
    d = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    knn = np.argsort(d, axis=1)[:, :k]
    for a, b in pairs:
        assert b in knn[a]
        assert labels[a] != labels[b]
    with pytest.raises(ValueError):
        knn_hard_negatives(z, 5, k=30, rng=SEED)


def test_hetero_contrastive_loss_finite():
    from spillety.embeddings.loss import hetero_contrastive_loss

    rng = np.random.default_rng(SEED)
    za = rng.normal(size=(10, 6))
    zt = rng.normal(size=(6, 6))
    pos = np.array([[i, i % 6] for i in range(10)])
    loss = hetero_contrastive_loss(za, zt, pos, tau=0.1)
    assert np.isfinite(loss) and loss > 0
    with pytest.raises(ValueError):
        hetero_contrastive_loss(za, zt, pos, tau=0.0)
    with pytest.raises(ValueError):
        hetero_contrastive_loss(za, zt, np.zeros((0, 2), dtype=int))


def test_distillation_loss_falls():
    from spillety.embeddings.distill import distill_teacher_to_student

    rng = np.random.default_rng(SEED)
    n = 40
    xl = rng.normal(size=(n, 6))
    xn = rng.normal(size=(n, 6)) * 0.5 + xl * 0.5
    teacher = rng.normal(size=(n, 2))
    _, hist = distill_teacher_to_student(
        teacher, xl, xn, hidden_dim=16, epochs=100, lr=0.1, seed=SEED
    )
    assert len(hist) == 100
    assert all(np.isfinite(hist))
    assert hist[-1] < hist[0]
