import numpy as np
import pytest

from spillety.embeddings.augment import (
    drop_edges,
    ks_gate,
    mask_features,
    perturb_features,
)
from spillety.embeddings.loss import anchor_loss, jaccard_index, nt_xent, pull_margin
from spillety.embeddings.pairs import (
    hard_negatives,
    positive_pairs,
    sample_negatives,
    sampling_probs,
)
from spillety.embeddings.temporal import encode_temporal_decoupled

SEED = 72


def _smoke_batch():
    g = np.random.default_rng(SEED)
    return g.normal(size=(32, 8))


def test_nt_xent_finite_and_decreases():
    z = _smoke_batch()
    loss_before = nt_xent(z, tau=0.1)
    assert np.isfinite(loss_before)
    improved = z.copy()
    for k in range(0, len(z), 2):
        mid = (z[k] + z[k + 1]) / 2
        improved[k] = 0.5 * z[k] + 0.5 * mid
        improved[k + 1] = 0.5 * z[k + 1] + 0.5 * mid
    assert nt_xent(improved, tau=0.1) < loss_before


def test_nt_xent_rejects_bad_tau():
    with pytest.raises(ValueError):
        nt_xent(_smoke_batch(), tau=0.0)


def test_pull_margin_from_jaccard():
    assert pull_margin(1.0, 0.0) == pytest.approx(1.0)
    assert pull_margin(1.0, 1.0) == pytest.approx(0.0)
    assert pull_margin(2.0, 0.3) == pytest.approx(1.4)
    assert jaccard_index({1, 2}, {2, 3}) == pytest.approx(1 / 3)


def test_anchor_loss_zero_when_compact_and_separated():
    g = np.random.default_rng(SEED)
    za = np.zeros((4, 4)) + g.normal(scale=1e-6, size=(4, 4))
    zn = np.full((6, 4), 10.0) + g.normal(scale=1e-6, size=(6, 4))
    assert anchor_loss(za, zn, lam=0.5, m0=1.0, jaccard=0.0) == pytest.approx(0.0)


def test_anchor_loss_grows_with_disagreement():
    g = np.random.default_rng(SEED)
    za = g.normal(scale=2.0, size=(6, 4))
    zn = g.normal(scale=2.0, size=(8, 4))
    low = anchor_loss(za, zn, lam=1.0, m0=1.0, jaccard=0.0)
    high = anchor_loss(za, zn, lam=1.0, m0=1.0, jaccard=1.0)
    assert high > low >= 0.0


def test_pair_counts_and_degree_correction():
    edges = np.array([[0, 1], [1, 2], [2, 3], [0, 3], [1, 3]])
    labels = np.array([0, 0, 1, 0])
    times = np.array([5, 5, 5, 6])
    pos = positive_pairs(edges, labels=labels, times=times, eps=1)
    assert len(pos) == 3
    assert positive_pairs(edges).shape == (5, 2)
    degrees = np.array([1, 5, 10, 50])
    p = sampling_probs(degrees)
    assert p.sum() == pytest.approx(1.0)
    assert p[0] > p[-1]
    neg = sample_negatives(4, degrees, 100, rng=SEED)
    assert neg.shape == (100, 2)
    assert (neg[:, 0] == 0).mean() > (neg[:, 0] == 3).mean()


def test_hard_negatives_curriculum_flag():
    labels = np.array([0, 0, 1, 1, 0, 1])
    times = np.array([3, 3, 3, 3, 9, 9])
    assert hard_negatives(labels, times, 10, use_curriculum=False).shape == (0, 2)
    hard = hard_negatives(labels, times, 10, use_curriculum=True, rng=SEED)
    assert len(hard) > 0
    assert np.all(labels[hard[:, 0]] != labels[hard[:, 1]])
    assert np.all(np.abs(times[hard[:, 0]] - times[hard[:, 1]]) <= 1)


def test_augment_rates_and_determinism():
    x = np.random.default_rng(SEED).normal(size=(50, 8))
    assert np.array_equal(mask_features(x, p=0.0, rng=SEED), x)
    assert (mask_features(x, p=1.0, rng=SEED) == 0.0).all()
    assert np.array_equal(mask_features(x, p=0.2, rng=SEED), mask_features(x, p=0.2, rng=SEED))
    edges = np.arange(40).reshape(2, 20)
    assert drop_edges(edges, p=0.0, rng=SEED).shape == (2, 20)
    assert drop_edges(edges, p=1.0, rng=SEED).shape == (2, 0)
    assert np.array_equal(
        perturb_features(x, rng=SEED), perturb_features(x, rng=SEED)
    )


def test_ks_gate_deterministic():
    g = np.random.default_rng(SEED)
    ref = g.normal(size=(200, 4))
    accept, p = ks_gate(ref, ref.copy())
    assert accept and p == pytest.approx(1.0)
    accept2, p2 = ks_gate(ref, ref + 5.0)
    assert not accept2 and p2 < 0.05
    assert ks_gate(ref, ref.copy()) == ks_gate(ref, ref.copy())


def test_sage_encode_shape():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from spillety.embeddings.sage import encode

    g = np.random.default_rng(SEED)
    x = g.normal(size=(12, 6)).astype(np.float32)
    edge_index = np.array([[i, (i + 1) % 12] for i in range(12)]).T
    z = encode(x, edge_index, hidden_dim=8, out_dim=4, seed=SEED)
    assert isinstance(z, torch.Tensor)
    assert z.shape == (12, 4)


def test_haar_level2_roundtrip():
    from spillety.features.wavelet import haar_level2

    x = np.random.default_rng(SEED).normal(size=(5, 16))
    wc = haar_level2(x)
    assert wc.shape == x.shape
    assert np.isfinite(wc).all()


def test_wavelet_features_doubles_columns():
    from spillety.features.wavelet import wavelet_features

    x = np.random.default_rng(SEED).normal(size=(8, 10))
    out = wavelet_features(x)
    assert out.shape == (8, 20)
    assert np.array_equal(out[:, :10], x)


def test_sage_encode_gated_shape():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from spillety.embeddings.sage import encode_gated

    g = np.random.default_rng(SEED)
    x = g.normal(size=(12, 6)).astype(np.float32)
    edge_index = np.array([[i, (i + 1) % 12] for i in range(12)]).T
    z = encode_gated(x, edge_index, hidden_dim=8, out_dim=4, seed=SEED)
    assert isinstance(z, torch.Tensor)
    assert z.shape == (12, 4)


def test_encode_temporal_decoupled_shape():
    torch = pytest.importorskip("torch")
    g = np.random.default_rng(SEED)
    n_nodes, in_dim, T = 10, 6, 4
    xs = [g.normal(size=(n_nodes, in_dim)).astype(np.float32) for _ in range(T)]
    edge_indices = [
        np.array([[i, (i + 1) % n_nodes] for i in range(n_nodes)]).T
        for _ in range(T)
    ]
    out = encode_temporal_decoupled(xs, edge_indices, hidden_dim=8, out_dim=4, seed=SEED)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (T, n_nodes, 4)


def test_haar_level2_constant_signal():
    from spillety.features.wavelet import haar_level2

    x = np.ones((3, 8)) * 5.0
    wc = haar_level2(x)
    assert wc.shape == x.shape
    assert np.isfinite(wc).all()


def test_haar_level2_odd_columns():
    from spillety.features.wavelet import haar_level2

    x = np.random.default_rng(SEED).normal(size=(4, 7))
    wc = haar_level2(x)
    assert wc.shape == x.shape
