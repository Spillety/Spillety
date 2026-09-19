from typing import Any


def _as_tensor(torch, a):
    import numpy as np

    if isinstance(a, torch.Tensor):
        return a.to(dtype=torch.float32)
    return torch.as_tensor(np.asarray(a), dtype=torch.float32)


class StudentMLP:
    """
    ## Two-layer MLP student on local plus neighbor-aggregated features (§4.8)

    Parameters
    ----------
    torch : module
        Imported torch module (kept explicit for CPU-only use).
    in_dim : int
        Concatenated feature width.
    hidden_dim : int
        Hidden width.
    out_dim : int
        Logit width, must match the teacher.
    """

    def __init__(self, torch, in_dim, hidden_dim=16, out_dim=2):
        self._torch = torch
        self.fc1 = torch.nn.Linear(in_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, out_dim)

    def __call__(self, x):
        torch = self._torch
        return self.fc2(torch.relu(self.fc1(x)))

    def parameters(self):
        return list(self.fc1.parameters()) + list(self.fc2.parameters())


def distill_teacher_to_student(
    teacher_logits,
    x_local,
    x_neigh,
    hidden_dim=16,
    epochs=200,
    lr=0.1,
    alpha=0.5,
    tau=2.0,
    seed=72,
) -> Any:
    """
    ## Full-batch distillation of teacher soft labels into an MLP student (§4.8)

    Parameters
    ----------
    teacher_logits : array-like
        Teacher logits (N, C).
    x_local : array-like
        Local node features (N, d1).
    x_neigh : array-like
        Precomputed neighbor aggregates (N, d2).
    hidden_dim : int
        Student hidden width.
    epochs : int
        Full-batch SGD steps.
    lr : float
        Learning rate, must be > 0.
    alpha : float
        Weight of the KL term vs the MSE term, in [0, 1].
    tau : float
        Softmax temperature, must be > 0.
    seed : int
        Seed for init and shuffling; CPU only.

    Returns
    ----------
    tuple[StudentMLP, list[float]]
        Trained student and per-epoch total loss history.
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError("torch is required for distillation") from e
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    if lr <= 0:
        raise ValueError(f"lr must be > 0, got {lr}")
    torch.manual_seed(seed)
    t = _as_tensor(torch, teacher_logits)
    xl = _as_tensor(torch, x_local)
    xn = _as_tensor(torch, x_neigh)
    if not (t.shape[0] == xl.shape[0] == xn.shape[0]):
        raise ValueError("teacher_logits, x_local, x_neigh must share N")
    x = torch.cat([xl, xn], dim=1)
    student = StudentMLP(torch, x.shape[1], hidden_dim, t.shape[1])
    opt = torch.optim.SGD(student.parameters(), lr=lr)
    soft = torch.softmax(t / tau, dim=1)
    # ponytail: full-batch SGD; switch to minibatches when N exceeds RAM.
    history = []
    for _ in range(epochs):
        opt.zero_grad()
        s = student(x)
        log_p = torch.log_softmax(s / tau, dim=1)
        kl = torch.nn.functional.kl_div(log_p, soft, reduction="batchmean") * tau * tau
        mse = ((s - t) ** 2).mean()
        loss = alpha * kl + (1 - alpha) * mse
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
    return student, history
