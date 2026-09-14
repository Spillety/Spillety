import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoder(nn.Module):
    """Pure PyTorch temporal convolution (NO PyG-Temporal).

    Temporal layers for message passing over time with self-attention masking.
    # ponytail: PyG-Temporal, add when temporal ops exceed pure PyTorch performance by >5%
    """

    def __init__(self, embed_dim: int = 128, num_layers: int = 2, kernel_size: int = 3) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(embed_dim, embed_dim, kernel_size, padding=kernel_size // 2)
            for _ in range(num_layers)
        ])
        self.temporal_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply temporal convolution with masking.

        Args:
            x: Tensor of shape (batch, seq, embed_dim)
            mask: Optional temporal mask (batch, seq)

        Returns:
            Encoded tensor of shape (batch, seq, embed_dim)
        """
        for conv in self.conv_layers:
            # Conv1d expects (batch, channels, seq)
            x_conv = conv(x.transpose(1, 2))
            x = x_conv.transpose(1, 2)
            x = F.relu(x)
        # Self-attention with key_padding_mask for temporal masking
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()
        out, _ = self.temporal_attn(x, x, x, key_padding_mask=key_padding_mask)
        return self.layer_norm(x + out)

    @property
    def temporal_depth(self) -> int:
        """Return number of temporal layers."""
        return self.num_layers


def demo() -> None:
    """Smoke test: TemporalEncoder with assert tests."""
    model = TemporalEncoder(embed_dim=64, num_layers=2)
    batch, seq = 2, 10
    x = torch.randn(batch, seq, 64)
    mask = torch.ones(batch, seq, dtype=torch.bool)
    out = model(x, mask)
    assert out.shape == (batch, seq, 64), f"Output shape mismatch: {out.shape}"
    assert not torch.isnan(out).any(), "Output must not contain NaN"
    assert model.temporal_depth == 2, "Temporal depth should be 2"
    # Test without mask
    out_no_mask = model(x)
    assert out_no_mask.shape == (batch, seq, 64), "No-mask output shape mismatch"
    print("TemporalEncoder demo passed")


if __name__ == "__main__":
    demo()
