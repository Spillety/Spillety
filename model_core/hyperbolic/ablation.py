import torch
from sklearn.metrics import roc_auc_score

from model_core.hyperbolic.message_passing import HyperbolicMessagePassing
from model_core.hyperbolic.poincare_ops import PoincareBallOps

GEOMETRIES = ("poincare", "euclidean", "lorentz")


def risk_scores(embeddings: torch.Tensor, scam_centroid: torch.Tensor, geometry: str, c: float = -1.0) -> torch.Tensor:
    """Negative distance to scam centroid: closer = riskier."""
    if geometry == "poincare":
        d = PoincareBallOps(dim=embeddings.shape[-1]).distance(embeddings, scam_centroid.expand_as(embeddings), c)
        return -d.squeeze(-1)
    return -torch.norm(embeddings - scam_centroid, dim=-1)


def ablation_compare(
    x: torch.Tensor,
    edges: torch.Tensor,
    labels: torch.Tensor,
    embed_dim: int | None = None,
    num_layers: int = 2,
    seed: int = 0,
    c: float = -1.0,
) -> dict:
    """Run each geometry forward pass and score AUC vs labels."""
    torch.manual_seed(seed)
    out: dict = {}
    for geometry in GEOMETRIES:
        dim = embed_dim or x.shape[-1]
        model = HyperbolicMessagePassing(embed_dim=dim, num_layers=num_layers, geometry=geometry)
        with torch.no_grad():
            emb = model(x, edges)
        scam_centroid = emb[labels == 1].mean(dim=0, keepdim=True)
        scores = risk_scores(emb, scam_centroid, geometry, c)
        y = labels.detach().cpu().numpy()
        s = scores.detach().cpu().numpy()
        auc = float(roc_auc_score(y, s)) if len(set(y.tolist())) == 2 else float("nan")
        out[geometry] = {"auc": auc, "scores": scores}
    return out
