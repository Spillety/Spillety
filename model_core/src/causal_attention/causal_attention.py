import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalAttention(nn.Module):
    """Backdoor-adjusted attention weights using is_exchange_internal as adjustment set member.

    Domain prior defines allowed edge types.
    # ponytail: prior strength — soft constraint weighted prior instead of hard mask
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_edge_types: int = 6,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_edge_types = num_edge_types
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.edge_proj = nn.Linear(num_edge_types, embed_dim)
        self.backdoor_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        edge_types: torch.Tensor,
        is_exchange_internal: torch.Tensor,
    ) -> torch.Tensor:
        """Compute backdoor-adjusted attention weights.

        Args:
            queries: Tensor of shape (batch, seq, embed_dim)
            keys: Tensor of shape (batch, seq, embed_dim)
            edge_types: Edge type indices (batch, seq, num_edge_types)
            is_exchange_internal: Confounder flag (batch, seq)

        Returns:
            Adjusted attention output tensor
        """
        # Project edge types into embedding space
        edge_emb = self.edge_proj(edge_types)
        # Adjust for backdoor confounder: is_exchange_internal
        confounder_adj = self.backdoor_proj(edge_emb) * is_exchange_internal.unsqueeze(-1).float()
        # Combine edge embedding with confounder adjustment
        adjusted_keys = keys + confounder_adj
        attn_out, _ = self.attention(queries, adjusted_keys, adjusted_keys)
        return attn_out

    def get_edge_mask(self, allowed_types: list[int]) -> torch.Tensor:
        """Create binary mask for allowed edge types from domain prior."""
        mask = torch.zeros(self.num_edge_types)
        for t in allowed_types:
            mask[t] = 1.0
        return mask


def demo() -> None:
    """Smoke test: CausalAttention with assert tests."""
    model = CausalAttention(embed_dim=64, num_heads=2, num_edge_types=6)
    batch, seq = 2, 8
    queries = torch.randn(batch, seq, 64)
    keys = torch.randn(batch, seq, 64)
    edge_types = torch.zeros(batch, seq, 6)
    is_exchange_internal = torch.ones(batch, seq)
    out = model(queries, keys, edge_types, is_exchange_internal)
    assert out.shape == (batch, seq, 64), f"Output shape mismatch: {out.shape}"
    assert not torch.isnan(out).any(), "Output must not contain NaN"
    mask = model.get_edge_mask([0, 1, 2])
    assert mask.sum() == 3.0, "Mask should have 3 allowed types"
    print("CausalAttention demo passed")


if __name__ == "__main__":
    demo()
