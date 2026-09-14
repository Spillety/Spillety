import torch
import pytest

from model_core.loss.focal_loss import FocalLoss
from model_core.loss.contrastive_loss import ContrastiveLoss
from model_core.loss.combined_loss import CombinedLoss
from model_core.loss.fsdp_wrapper import init_fsdp, cleanup_fsdp


def test_focal_loss_confident():
    loss_fn = FocalLoss(gamma=2.0)
    logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
    loss = loss_fn(logits, torch.tensor([0, 1]))
    assert loss.item() < 0.01


def test_focal_loss_floor():
    loss_fn = FocalLoss(gamma=2.0)
    logits = torch.tensor([[0.1, -0.1]])
    loss = loss_fn(logits, torch.tensor([0]))
    assert loss.item() >= 0.1


def test_focal_loss_shape():
    loss_fn = FocalLoss(gamma=2.0)
    logits = torch.randn(16, 6)
    targets = torch.randint(0, 6, (16,))
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_contrastive_low_tau():
    loss_low = ContrastiveLoss(tau=0.05)
    loss_high = ContrastiveLoss(tau=0.5)
    torch.manual_seed(0)
    a = torch.randn(32, 128)
    p = a + 0.01 * torch.randn(32, 128)
    # Lower tau sharpens the softmax, so near-duplicate positives yield lower loss.
    assert loss_low(a, p).item() < loss_high(a, p).item()


def test_contrastive_shape():
    loss_fn = ContrastiveLoss(tau=0.1)
    a = torch.randn(16, 128)
    p = a + 0.01 * torch.randn(16, 128)
    loss = loss_fn(a, p)
    assert loss.dim() == 0


def test_contrastive_similarity():
    loss_fn = ContrastiveLoss(tau=0.1)
    a = torch.randn(32, 128)
    p = a.clone()
    loss = loss_fn(a, p)
    assert loss.item() >= 0


def test_combined_loss_beta_init():
    loss_fn = CombinedLoss(alpha=1.0, beta_init=0.1, beta_final=0.5, annealing_epochs=50)
    assert loss_fn.current_beta == 0.1


def test_combined_loss_beta_annealing():
    loss_fn = CombinedLoss(alpha=1.0, beta_init=0.1, beta_final=0.5, annealing_epochs=50)
    loss_fn.update_beta(25)
    assert 0.1 <= loss_fn.current_beta <= 0.5


def test_combined_loss_beta_final():
    loss_fn = CombinedLoss(alpha=1.0, beta_init=0.1, beta_final=0.5, annealing_epochs=50)
    loss_fn.update_beta(100)
    assert abs(loss_fn.current_beta - 0.5) < 1e-6


def test_combined_loss_forward():
    loss_fn = CombinedLoss(alpha=1.0, beta_init=0.1, beta_final=0.5, annealing_epochs=50)
    logits = torch.randn(32, 6)
    targets = torch.randint(0, 6, (32,))
    anchor = torch.randn(32, 128)
    positive = anchor + 0.01 * torch.randn(32, 128)
    loss = loss_fn(logits, targets, anchor, positive)
    assert loss.item() > 0


def test_fsdp_init():
    import torch.nn as nn
    import unittest.mock as mock
    with mock.patch("model_core.loss.fsdp_wrapper.torch.cuda.is_available", return_value=True), \
         mock.patch("model_core.loss.fsdp_wrapper.torch.cuda.current_device", return_value=0), \
         mock.patch("torch.distributed.init_process_group"), \
         mock.patch("model_core.loss.fsdp_wrapper.FSDP", side_effect=lambda m, **kw: m):
        model = nn.Linear(128, 64)
        wrapped = init_fsdp(model)
        assert wrapped is model
        cleanup_fsdp()


def test_fsdp_requires_gpu():
    import torch.nn as nn
    import unittest.mock as mock
    with mock.patch("model_core.loss.fsdp_wrapper.torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="requires GPU"):
            init_fsdp(nn.Linear(8, 4))
