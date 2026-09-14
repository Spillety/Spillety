import torch
import torch.nn as nn
from collections.abc import Mapping, Sequence


class CausalAttention(nn.Module):
    """SCM-guided attention: P(Y|do(X)) = sum_z P(Y|X,Z=z)P(Z=z) over strata.

    # ponytail: strata are quantile buckets of confounders, not a learned propensity model
    """

    CONTINUOUS_CONFOUNDERS = ("amount", "time_of_day", "address_features")

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_edge_types: int = 6,
        max_strata: int = 8,
        stratum_bins: int = 4,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_edge_types = num_edge_types
        self.max_strata = max_strata
        self.stratum_bins = stratum_bins
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.edge_proj = nn.Linear(num_edge_types, embed_dim)
        self.backdoor_proj = nn.Linear(embed_dim, embed_dim)
        self.stratum_embed = nn.Embedding(max_strata, embed_dim)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        edge_types: torch.Tensor,
        is_exchange_internal: torch.Tensor,
        confounders: Mapping[str, torch.Tensor] | torch.Tensor | None = None,
        allowed_types: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Backdoor-adjusted attention: sum_z P(Y|X,Z=z)P(Z=z) over strata."""
        mask = self._resolve_mask(allowed_types, edge_types.device)
        edge_emb = self.edge_proj(edge_types.float() * mask)
        base_keys = keys + edge_emb

        strata, weights = self.stratify_confounders(
            confounders, is_exchange_internal, queries.shape[:2], queries.device
        )
        out = torch.zeros_like(queries)
        for s in range(len(weights)):
            w = weights[s]
            if w.item() == 0.0:
                continue
            # One stratum value broadcast to all positions: breaks X<-Z dependence (do-operator).
            bias = self.backdoor_proj(self.stratum_embed.weight[s]).view(1, 1, -1)
            keys_s = base_keys + bias
            attn_s, _ = self.attention(queries, keys_s, keys_s)
            out = out + w * attn_s
        return out

    def stratify_confounders(
        self,
        confounders: Mapping[str, torch.Tensor] | torch.Tensor | None,
        is_exchange_internal: torch.Tensor,
        batch_shape: torch.Size | tuple[int, int],
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bucketize confounders into stratum ids with empirical P(Z) weights."""
        items = self._confounder_items(confounders, is_exchange_internal)
        codes: list[torch.Tensor] = []
        for name, t in items:
            t = torch.as_tensor(t, device=device)
            if t.dim() == 3:
                # ponytail: multi-dim address features reduced by mean before bucketing
                t = t.float().mean(dim=-1)
            t = t.reshape(-1)
            if t.is_floating_point() or name in self.CONTINUOUS_CONFOUNDERS:
                codes.append(self._discretize(t, self.stratum_bins))
            else:
                _, inv = torch.unique(t.long(), return_inverse=True)
                codes.append(inv)
        combined = codes[0]
        for c in codes[1:]:
            combined = combined * (int(c.max().item()) + 1) + c
        _, compact = torch.unique(combined, return_inverse=True)
        n = int(compact.max().item()) + 1
        counts = torch.bincount(compact, minlength=n).float()
        weights = counts / counts.sum()
        if n > self.max_strata:
            # ponytail: rare strata merged into one bucket to bound O(S) attention calls
            order = torch.argsort(counts, descending=True)
            remap = torch.full((n,), self.max_strata - 1, dtype=torch.long)
            remap[order[: self.max_strata - 1]] = torch.arange(self.max_strata - 1)
            compact = remap[compact]
            counts = torch.bincount(compact, minlength=self.max_strata).float()
            weights = counts / counts.sum()
        b, s = batch_shape[0], batch_shape[1]
        return compact.reshape(b, s), weights

    @staticmethod
    def backdoor_effect(
        outcome: torch.Tensor, treatment: torch.Tensor, strata: torch.Tensor
    ) -> torch.Tensor:
        """Stratified causal effect E_Z[E[Y|X=1,Z]-E[Y|X=0,Z]]."""
        y = outcome.float().reshape(-1)
        x = treatment.float().reshape(-1)
        z = strata.long().reshape(-1)
        counts = torch.bincount(z).float()
        probs = counts / counts.sum()
        effect = torch.zeros((), dtype=torch.float32)
        mass = torch.zeros(())
        for s in range(len(counts)):
            sel = z == s
            y1 = y[sel & (x > 0.5)]
            y0 = y[sel & (x <= 0.5)]
            if y1.numel() == 0 or y0.numel() == 0:
                continue
            effect = effect + probs[s] * (y1.mean() - y0.mean())
            mass = mass + probs[s]
        return effect / mass.clamp_min(1e-8)

    def edge_causal_effects(
        self,
        edge_labels: Sequence[str],
        treatment: torch.Tensor,
        outcome: torch.Tensor,
        strata: torch.Tensor,
    ) -> list[dict]:
        """Per-edge effects as explanation.causal_path entries.

        Returns [{"edge": "src→dst", "causal_effect": float}]; attach each
        value as nx edge attr for CausalPathFinder or straight into alert JSON.
        """
        t = treatment.float()
        assert t.dim() == 2 and t.shape[1] == len(edge_labels), "treatment must be (N, E)"
        return [
            {"edge": label, "causal_effect": float(self.backdoor_effect(outcome, t[:, j], strata))}
            for j, label in enumerate(edge_labels)
        ]

    def get_edge_mask(self, allowed_types: list[int]) -> torch.Tensor:
        """Create binary mask for allowed edge types from domain prior."""
        mask = torch.zeros(self.num_edge_types)
        for t in allowed_types:
            mask[t] = 1.0
        return mask

    def _resolve_mask(
        self, allowed_types: Sequence[int] | torch.Tensor | None, device: torch.device | str
    ) -> torch.Tensor:
        if allowed_types is None:
            return torch.ones(self.num_edge_types, device=device)
        if isinstance(allowed_types, torch.Tensor):
            return allowed_types.float().to(device)
        return self.get_edge_mask(list(allowed_types)).to(device)

    def _confounder_items(
        self,
        confounders: Mapping[str, torch.Tensor] | torch.Tensor | None,
        is_exchange_internal: torch.Tensor,
    ) -> list[tuple[str, torch.Tensor]]:
        if confounders is None:
            return [("is_exchange_internal", is_exchange_internal)]
        if isinstance(confounders, torch.Tensor):
            return [("is_exchange_internal", is_exchange_internal), ("extra", confounders)]
        items = list(confounders.items())
        if "is_exchange_internal" not in confounders:
            items.append(("is_exchange_internal", is_exchange_internal))
        return items

    @staticmethod
    def _discretize(t: torch.Tensor, bins: int) -> torch.Tensor:
        if bins <= 1 or bool((t == t[0]).all()):
            return torch.zeros(t.numel(), dtype=torch.long, device=t.device)
        cuts = torch.quantile(t.float(), torch.linspace(0, 1, bins + 1, device=t.device)[1:-1])
        return torch.bucketize(t.float(), cuts.clamp_min(1e-8)).long()
