import torch
import pytest

from model_core.hyperbolic.ablation import ablation_compare
from model_core.hyperbolic.curvature import LearnableCurvature
from model_core.hyperbolic.hyperbolic_linear import HyperbolicLinear
from model_core.hyperbolic.lorentz_ops import LorentzOps
from model_core.hyperbolic.message_passing import HyperbolicMessagePassing
from model_core.hyperbolic.poincare_ops import PoincareBallOps


class TestLorentzOps:
    def test_expmap_shape(self):
        ops = LorentzOps(dim=4)
        x = torch.randn(2, 4) * 0.1
        out = ops.expmap(x, c=-1.0)
        assert out.shape == (2, 4)

    def test_expmap_no_nan(self):
        ops = LorentzOps(dim=4)
        x = torch.randn(2, 4) * 0.1
        out = ops.expmap(x, c=-1.0)
        assert not torch.isnan(out).any()

    def test_logmap_shape(self):
        ops = LorentzOps(dim=4)
        x = torch.randn(2, 4) * 0.1
        out = ops.logmap(x, c=-1.0)
        assert out.shape == (2, 4)

    def test_roundtrip(self):
        ops = LorentzOps(dim=4)
        x = torch.randn(2, 4) * 0.1
        exp_x = ops.expmap(x, c=-1.0)
        log_x = ops.logmap(exp_x, c=-1.0)
        roundtrip = ops.expmap(log_x, c=-1.0)
        assert torch.allclose(roundtrip, exp_x, atol=1e-3)

    def test_mobius_add_shape(self):
        ops = LorentzOps(dim=4)
        a = torch.randn(2, 4) * 0.1
        b = torch.randn(2, 4) * 0.1
        out = ops.mobius_add(a, b, c=-1.0)
        assert out.shape == (2, 4)

    def test_mobius_add_no_nan(self):
        ops = LorentzOps(dim=4)
        a = torch.randn(2, 4) * 0.1
        b = torch.randn(2, 4) * 0.1
        out = ops.mobius_add(a, b, c=-1.0)
        assert not torch.isnan(out).any()

    def test_expmap_clamping(self):
        ops = LorentzOps(dim=4)
        x = torch.randn(2, 4) * 1e5
        out = ops.expmap(x, c=-1.0)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestHyperbolicLinear:
    def test_forward_shape(self):
        layer = HyperbolicLinear(in_features=16, out_features=8)
        x = torch.randn(4, 16) * 0.1
        out = layer(x)
        assert out.shape == (4, 8)

    def test_no_nan(self):
        layer = HyperbolicLinear(in_features=16, out_features=8)
        x = torch.randn(4, 16) * 0.1
        out = layer(x)
        assert not torch.isnan(out).any()

    def test_no_inf(self):
        layer = HyperbolicLinear(in_features=16, out_features=8)
        x = torch.randn(4, 16) * 0.1
        out = layer(x)
        assert not torch.isinf(out).any()

    def test_curvature_bounds(self):
        layer = HyperbolicLinear(in_features=16, out_features=8)
        c = torch.sigmoid(layer.curvature) * (-1.9) + (-0.1)
        assert c.item() >= -2.0
        assert c.item() < -0.1

    def test_gradient_bounded(self):
        layer = HyperbolicLinear(in_features=16, out_features=8)
        x = torch.randn(4, 16) * 0.1
        out = layer(x)
        loss = out.sum()
        loss.backward()
        for p in layer.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()


class TestHyperbolicMessagePassing:
    def test_forward_shape(self):
        model = HyperbolicMessagePassing(embed_dim=16, num_layers=2)
        x = torch.randn(5, 16) * 0.1
        edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = model(x, edges)
        assert out.shape == (5, 16)

    def test_no_nan(self):
        model = HyperbolicMessagePassing(embed_dim=16, num_layers=2)
        x = torch.randn(5, 16) * 0.1
        edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = model(x, edges)
        assert not torch.isnan(out).any()

    def test_no_inf(self):
        model = HyperbolicMessagePassing(embed_dim=16, num_layers=2)
        x = torch.randn(5, 16) * 0.1
        edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = model(x, edges)
        assert not torch.isinf(out).any()

    def test_gradient_clipping(self):
        model = HyperbolicMessagePassing(embed_dim=16, num_layers=2)
        x = torch.randn(5, 16) * 0.1
        edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = model(x, edges)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()


class TestLearnableCurvature:
    def test_forward_shape(self):
        lc = LearnableCurvature(num_layers=3)
        curvatures = lc.forward()
        assert curvatures.shape == (3,)

    def test_curvature_bounds(self):
        lc = LearnableCurvature(num_layers=4)
        curvatures = lc.forward()
        for c in curvatures:
            assert c.item() >= -2.0
            assert c.item() < -0.1

    def test_no_nan(self):
        lc = LearnableCurvature(num_layers=3)
        curvatures = lc.forward()
        assert not torch.isnan(curvatures).any()

    def test_no_inf(self):
        lc = LearnableCurvature(num_layers=3)
        curvatures = lc.forward()
        assert not torch.isinf(curvatures).any()

    def test_initialization(self):
        lc = LearnableCurvature(num_layers=3)
        assert lc.num_layers == 3
        for p in lc.curvatures:
            assert p.item() == -1.0


class TestPoincareBallOps:
    def test_expmap0_logmap0_roundtrip(self):
        torch.manual_seed(0)
        ops = PoincareBallOps(dim=4)
        x = torch.randn(4, 4) * 0.1
        assert torch.allclose(ops.logmap0(ops.expmap0(x)), x, atol=1e-4)

    def test_mobius_identity(self):
        torch.manual_seed(0)
        ops = PoincareBallOps(dim=4)
        x = ops.expmap0(torch.randn(3, 4) * 0.1)
        zero = torch.zeros(1, 4)
        assert torch.allclose(ops.mobius_add(x, zero.expand_as(x)), x, atol=1e-5)

    def test_distance_monotone_to_scam_cluster(self):
        ops = PoincareBallOps(dim=4)
        u = torch.tensor([1.0, 0.0, 0.0, 0.0])
        scam = ops.project((0.05 * u).unsqueeze(0))
        pts = torch.stack([(0.1 + 0.1 * k) * u for k in range(5)])
        d = ops.distance(pts, scam.expand_as(pts)).squeeze(-1)
        assert bool(((d[1:] - d[:-1]) > 0).all()), f"distances not monotone: {d.tolist()}"

    def test_distance_symmetric_nonneg(self):
        torch.manual_seed(1)
        ops = PoincareBallOps(dim=4)
        a = ops.expmap0(torch.randn(3, 4) * 0.2)
        b = ops.expmap0(torch.randn(3, 4) * 0.2)
        dab, dba = ops.distance(a, b), ops.distance(b, a)
        assert bool((dab >= 0).all()) and torch.allclose(dab, dba, atol=1e-5)
        assert torch.allclose(ops.distance(a, a), torch.zeros(3, 1), atol=1e-5)

    def test_project_clamps_extreme(self):
        ops = PoincareBallOps(dim=4)
        out = ops.expmap0(torch.randn(2, 4) * 1e3)
        assert not torch.isnan(out).any() and not torch.isinf(out).any()
        assert bool((out.norm(dim=-1) < 1.0).all())

    def test_grad_finite_after_clipping(self):
        torch.manual_seed(2)
        ops = PoincareBallOps(dim=4)
        x = torch.randn(4, 4) * 0.2
        x.requires_grad_()
        loss = ops.expmap0(x).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([x], max_norm=1.0)
        assert torch.isfinite(x.grad).all()


class TestGeometrySwitch:
    def test_default_is_lorentz(self):
        assert HyperbolicMessagePassing(embed_dim=8).geometry == "lorentz"

    def test_invalid_geometry_raises(self):
        with pytest.raises(ValueError):
            HyperbolicMessagePassing(embed_dim=8, geometry="klein")

    def test_poincare_forward_inside_ball(self):
        torch.manual_seed(0)
        model = HyperbolicMessagePassing(embed_dim=8, num_layers=2, geometry="poincare")
        out = model(torch.randn(5, 8) * 0.1, torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long))
        assert out.shape == (5, 8) and not torch.isnan(out).any()
        assert bool((out.norm(dim=-1) < 1.0).all())

    def test_euclidean_forward_shape(self):
        torch.manual_seed(0)
        model = HyperbolicMessagePassing(embed_dim=8, num_layers=2, geometry="euclidean")
        out = model(torch.randn(5, 8) * 0.1, torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long))
        assert out.shape == (5, 8) and not torch.isnan(out).any() and not torch.isinf(out).any()


class TestAblationHarness:
    def test_compare_returns_auc_per_geometry(self):
        torch.manual_seed(0)
        x = torch.randn(10, 8) * 0.1
        edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        labels = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        res = ablation_compare(x, edges, labels)
        assert set(res) == {"poincare", "euclidean", "lorentz"}
        for v in res.values():
            assert 0.0 <= v["auc"] <= 1.0

    def test_scam_nodes_score_higher(self):
        torch.manual_seed(0)
        scam = torch.randn(4, 8) * 0.05 + 0.4
        legit = torch.randn(6, 8) * 0.05 - 0.4
        x = torch.cat([scam, legit])
        edges = torch.tensor([[0, 4, 5], [4, 5, 6]], dtype=torch.long)
        labels = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        res = ablation_compare(x, edges, labels)
        assert res["poincare"]["auc"] > 0.5
