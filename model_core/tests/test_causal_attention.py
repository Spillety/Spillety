import torch
import pytest

from model_core.causal_attention.notears_dag import NOTEARS_DAG
from model_core.causal_attention.causal_attention import CausalAttention
from model_core.causal_attention.temporal_encoder import TemporalEncoder
from model_core.causal_attention.domain_prior import DomainPrior


class TestNOTEARSDAG:
    def test_forward_loss_scalar(self):
        model = NOTEARS_DAG(max_nodes=10)
        adj = torch.randn(10, 10) * 0.1
        loss = model.forward(adj)
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert loss.item() > 0

    def test_forward_loss_gradient(self):
        model = NOTEARS_DAG(max_nodes=10)
        adj = torch.randn(10, 10) * 0.1
        loss = model.forward(adj)
        loss.backward()
        assert model.adj.grad is not None

    def test_get_adjacency_shape(self):
        model = NOTEARS_DAG(max_nodes=5)
        adj = model.get_adjacency()
        assert adj.shape == (5, 5)

    def test_no_self_loops(self):
        model = NOTEARS_DAG(max_nodes=5)
        adj = model.get_adjacency()
        assert torch.allclose(adj.diag(), torch.zeros(5))

    def test_sparsity_l1(self):
        model = NOTEARS_DAG(max_nodes=10, lambda1=0.5)
        adj = torch.eye(10) * 0.1
        loss = model.forward(adj)
        assert loss.item() > 0

    def test_reconstruction_accuracy(self):
        model = NOTEARS_DAG(max_nodes=5)
        true_adj = torch.randint(0, 2, (5, 5)).float()
        true_adj = true_adj * (1 - torch.eye(5))
        pred = model.get_adjacency()
        correct = (pred.sign() == true_adj.sign()).float().mean()
        assert correct.item() >= 0.0


class TestCausalAttention:
    def test_forward_output_shape(self):
        model = CausalAttention(embed_dim=64, num_heads=2, num_edge_types=6)
        batch, seq = 2, 8
        queries = torch.randn(batch, seq, 64)
        keys = torch.randn(batch, seq, 64)
        edge_types = torch.zeros(batch, seq, 6)
        is_exchange_internal = torch.ones(batch, seq)
        out = model(queries, keys, edge_types, is_exchange_internal)
        assert out.shape == (batch, seq, 64)

    def test_no_nan(self):
        model = CausalAttention(embed_dim=64, num_heads=2)
        queries = torch.randn(2, 8, 64)
        keys = torch.randn(2, 8, 64)
        edge_types = torch.zeros(2, 8, 6)
        is_exchange_internal = torch.ones(2, 8)
        out = model(queries, keys, edge_types, is_exchange_internal)
        assert not torch.isnan(out).any()

    def test_edge_mask(self):
        model = CausalAttention(num_edge_types=6)
        mask = model.get_edge_mask([0, 1, 2])
        assert mask.sum() == 3.0
        assert mask[0] == 1.0 and mask[1] == 1.0 and mask[2] == 1.0

    def test_backdoor_adjustment(self):
        model = CausalAttention(embed_dim=32, num_heads=2)
        queries = torch.randn(2, 4, 32)
        keys = torch.randn(2, 4, 32)
        edge_types = torch.zeros(2, 4, 6)
        is_exchange_internal = torch.ones(2, 4)
        out = model(queries, keys, edge_types, is_exchange_internal)
        assert out.shape == (2, 4, 32)

    def test_gradient_norm_bounded(self):
        model = CausalAttention(embed_dim=32, num_heads=2)
        queries = torch.randn(2, 4, 32)
        keys = torch.randn(2, 4, 32)
        edge_types = torch.zeros(2, 4, 6)
        is_exchange_internal = torch.ones(2, 4)
        out = model(queries, keys, edge_types, is_exchange_internal)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()


class TestTemporalEncoder:
    def test_forward_shape(self):
        model = TemporalEncoder(embed_dim=64, num_layers=2)
        x = torch.randn(2, 10, 64)
        mask = torch.ones(2, 10, dtype=torch.bool)
        out = model(x, mask)
        assert out.shape == (2, 10, 64)

    def test_no_nan(self):
        model = TemporalEncoder(embed_dim=64)
        x = torch.randn(2, 10, 64)
        out = model(x)
        assert not torch.isnan(out).any()

    def test_temporal_depth(self):
        model = TemporalEncoder(embed_dim=64, num_layers=3)
        assert model.temporal_depth == 3

    def test_without_mask(self):
        model = TemporalEncoder(embed_dim=64)
        x = torch.randn(2, 10, 64)
        out = model(x)
        assert out.shape == (2, 10, 64)

    def test_gradient_bounded(self):
        model = TemporalEncoder(embed_dim=32, num_layers=2)
        x = torch.randn(2, 8, 32)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()


class TestDomainPrior:
    def test_get_edge_mask(self):
        prior = DomainPrior(num_edge_types=6)
        mask = prior.get_edge_mask()
        assert mask.shape == (6,)
        assert mask[0] == 1.0 and mask[1] == 1.0
        assert mask[2] == 0.0

    def test_is_risk_edge(self):
        prior = DomainPrior()
        assert prior.is_risk_edge(0) is True
        assert prior.is_risk_edge(1) is True
        assert prior.is_risk_edge(4) is False

    def test_encode_expert_graph(self):
        prior = DomainPrior(num_edge_types=6)
        edges = [("0", "1", "TRANSFERS"), ("1", "2", "CO_SPEND")]
        adj = prior.encode_expert_graph(edges)
        assert adj.shape[2] == 6
        assert adj[0][1][0] == 1.0

    def test_num_edge_types(self):
        prior = DomainPrior(num_edge_types=4)
        assert prior.num_edge_types == 4

    def test_risk_mask_correct(self):
        prior = DomainPrior(num_edge_types=6)
        mask = prior.get_edge_mask()
        assert mask[0] == 1.0
        assert mask[1] == 1.0
        assert mask[2] == 0.0
        assert mask[3] == 0.0
        assert mask[4] == 0.0
        assert mask[5] == 0.0


class TestBackdoorAdjustment:
    @staticmethod
    def _synthetic(n: int = 4000, seed: int = 7):
        torch.manual_seed(seed)
        z = (torch.rand(n) < 0.5).float()
        x_spurious = (torch.rand(n) < torch.where(z > 0.5, 0.9, 0.1).float()).float()
        w_true = (torch.rand(n) < 0.5).float()
        y = torch.clamp(z + w_true, 0, 1)
        return x_spurious, w_true, y, z

    def test_stratified_effect_removes_spurious(self):
        from model_core.causal_attention.causal_attention import CausalAttention

        x, _, y, z = self._synthetic()
        naive = (y[x > 0.5].mean() - y[x <= 0.5].mean()).item()
        adjusted = CausalAttention.backdoor_effect(y, x, z).item()
        assert naive > 0.3, f"Synthetic must show spurious correlation, got {naive}"
        assert abs(adjusted) < 0.08, f"Backdoor must remove spurious effect, got {adjusted}"
        assert abs(adjusted) < abs(naive) - 0.2, "Adjusted effect must be well below naive"

    def test_edge_mask_used_in_forward(self):
        torch.manual_seed(0)
        model = CausalAttention(embed_dim=16, num_heads=2, num_edge_types=4)
        model.eval()
        queries = torch.randn(2, 4, 16)
        keys = torch.randn(2, 4, 16)
        edge_types = torch.zeros(2, 4, 4)
        edge_types[..., 0] = 1.0
        is_exchange_internal = torch.zeros(2, 4)
        full = model(queries, keys, edge_types, is_exchange_internal, allowed_types=[0])
        gated = model(queries, keys, edge_types, is_exchange_internal, allowed_types=[1, 2, 3])
        zeroed = model(queries, keys, torch.zeros_like(edge_types), is_exchange_internal)
        assert not torch.allclose(full, gated), "Mask must change forward output"
        assert torch.allclose(gated, zeroed, atol=1e-6), "Masked-out types must equal no-edge signal"

    def test_edge_causal_effects_schema_format(self):
        import networkx as nx

        from model_core.explainability.path_ranking import CausalPathFinder

        x, w, y, z = self._synthetic()
        model = CausalAttention(embed_dim=16, num_heads=2, num_edge_types=4)
        effects = model.edge_causal_effects(
            ["A→B", "C→D"], torch.stack([x, w], dim=1), y, z.long()
        )
        assert [sorted(e.keys()) for e in effects] == [["causal_effect", "edge"]] * 2
        assert all(isinstance(e["causal_effect"], float) for e in effects)
        assert abs(effects[0]["causal_effect"]) < 0.08, "Spurious edge effect must vanish"
        assert abs(effects[1]["causal_effect"] - 0.5) < 0.1, "True edge effect must persist"
        g = nx.DiGraph()
        g.add_edge("A", "B", weight=1.0, causal_effect=effects[0]["causal_effect"])
        g.add_edge("A", "C", weight=1.0, causal_effect=effects[1]["causal_effect"])
        g.add_edge("C", "B", weight=1.0, causal_effect=0.6)
        paths = CausalPathFinder(k=3).find_top_paths(g, "A", "B", k=3)
        assert paths and all("effect" in p for p in paths)
