"""
Transformer De-obfuscation Model
=================================
A Transformer encoder-decoder that takes tokenized obfuscated Python code
(augmented with graph spectral features) and outputs clean, readable code.

Architecture:
  - RoPE (Rotary Position Embeddings) for position encoding
  - SwiGLU feed-forward layers (modern, efficient activation)
  - RMSNorm for layer normalization
  - Cross-attention between encoder (obfuscated) and decoder (clean)
  - Graph feature injection via learned projection

Model size: ~10M parameters (trainable on a single GPU in hours).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# Building Blocks
# ──────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x * rms


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) from Su et al., 2021.

    Encodes position information directly into the attention mechanism
    by rotating query and key vectors in 2D subspaces.
    """

    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary embeddings to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, seq_len, num_heads, head_dim)
        """
        _, seq_len, _, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).float().unsqueeze(1)
        freqs = positions * self.inv_freq.to(x.device)

        sin_f = freqs.sin()
        cos_f = freqs.cos()

        x0, x1 = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack([
            x0 * cos_f - x1 * sin_f,
            x0 * sin_f + x1 * cos_f,
        ], dim=-1).flatten(-2)

        return rotated


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network (Shazeer, 2020).

    Uses SiLU (Swish) gating for improved gradient flow
    compared to standard ReLU FFNs.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

        # Orthogonal initialization for stable training
        for layer in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.orthogonal_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with RoPE.

    Supports both self-attention and cross-attention.
    """

    def __init__(self, dim: int, num_heads: int, is_cross_attention: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.rope = RotaryPositionEmbedding(self.head_dim)
        self.is_cross_attention = is_cross_attention

        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.orthogonal_(proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        q = self.rope(q).transpose(1, 2)

        kv_input = context if self.is_cross_attention and context is not None else x
        kv_len = kv_input.shape[1]

        k = self.k_proj(kv_input).view(batch, kv_len, self.num_heads, self.head_dim)
        v = self.v_proj(kv_input).view(batch, kv_len, self.num_heads, self.head_dim)

        if not self.is_cross_attention:
            k = self.rope(k)

        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0
        )

        output = attn_output.transpose(1, 2).reshape(batch, seq_len, self.dim)
        return self.o_proj(output)


# ──────────────────────────────────────────────────────────────
# Transformer Blocks
# ──────────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    """Transformer encoder block: self-attention + SwiGLU FFN."""

    def __init__(self, dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads)
        self.ffn = SwiGLU(dim, ffn_dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.ffn(self.norm2(x))
        return x


class DecoderBlock(nn.Module):
    """
    Transformer decoder block:
      self-attention + cross-attention + SwiGLU FFN.
    """

    def __init__(self, dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, num_heads)
        self.cross_attn = MultiHeadAttention(dim, num_heads, is_cross_attention=True)
        self.ffn = SwiGLU(dim, ffn_dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.norm3 = RMSNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ):
        x = x + self.self_attn(self.norm1(x), mask=causal_mask)
        x = x + self.cross_attn(self.norm2(x), context=encoder_output)
        x = x + self.ffn(self.norm3(x))
        return x


# ──────────────────────────────────────────────────────────────
# Full De-obfuscation Model
# ──────────────────────────────────────────────────────────────

class DeobfuscatorModel(nn.Module):
    """
    Structure-Aware Neural De-obfuscator.

    An encoder-decoder Transformer that takes obfuscated Python source
    tokens + graph spectral features and produces clean source tokens.

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary (default: 256, byte-level).
    dim : int
        Model dimension (default: 256).
    num_heads : int
        Number of attention heads (default: 8).
    ffn_dim : int
        FFN hidden dimension (default: 1024).
    num_encoder_layers : int
        Number of encoder blocks (default: 6).
    num_decoder_layers : int
        Number of decoder blocks (default: 6).
    graph_feature_dim : int
        Dimensionality of the graph feature vector (default: 130).
    max_seq_len : int
        Maximum sequence length (default: 2048).
    """

    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        graph_feature_dim: int = 130,
        max_seq_len: int = 2048,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Token embeddings
        self.encoder_embedding = nn.Embedding(vocab_size, dim)
        self.decoder_embedding = nn.Embedding(vocab_size, dim)

        # Graph feature projection → injected into encoder
        self.graph_proj = nn.Sequential(
            nn.Linear(graph_feature_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # Encoder stack
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(dim, num_heads, ffn_dim)
            for _ in range(num_encoder_layers)
        ])
        self.encoder_norm = RMSNorm(dim)

        # Decoder stack
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(dim, num_heads, ffn_dim)
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = RMSNorm(dim)

        # Output head
        self.output_proj = nn.Linear(dim, vocab_size, bias=False)

        # Tie encoder/decoder embeddings for parameter efficiency
        self.decoder_embedding.weight = self.encoder_embedding.weight
        self.output_proj.weight = self.encoder_embedding.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with scaled normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create a causal (lower-triangular) attention mask."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        mask = mask.masked_fill(mask == 0, float("-inf"))
        mask = mask.masked_fill(mask == 1, 0.0)
        return mask.unsqueeze(0).unsqueeze(0)

    def encode(
        self,
        src_tokens: torch.Tensor,
        graph_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode obfuscated source tokens.

        Parameters
        ----------
        src_tokens : torch.Tensor
            (batch, src_len) byte-level token IDs.
        graph_features : torch.Tensor, optional
            (batch, graph_feature_dim) spectral graph features.

        Returns
        -------
        torch.Tensor
            (batch, src_len, dim) encoder hidden states.
        """
        x = self.encoder_embedding(src_tokens)

        # Inject graph features as a learnable bias
        if graph_features is not None:
            graph_emb = self.graph_proj(graph_features)  # (batch, dim)
            x = x + graph_emb.unsqueeze(1)  # Broadcast across sequence

        for layer in self.encoder_layers:
            x = layer(x)

        return self.encoder_norm(x)

    def decode(
        self,
        tgt_tokens: torch.Tensor,
        encoder_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode clean source tokens given encoder context.

        Parameters
        ----------
        tgt_tokens : torch.Tensor
            (batch, tgt_len) target token IDs.
        encoder_output : torch.Tensor
            (batch, src_len, dim) encoder hidden states.

        Returns
        -------
        torch.Tensor
            (batch, tgt_len, vocab_size) logits.
        """
        tgt_len = tgt_tokens.shape[1]
        causal_mask = self._make_causal_mask(tgt_len, tgt_tokens.device)

        x = self.decoder_embedding(tgt_tokens)

        for layer in self.decoder_layers:
            x = layer(x, encoder_output, causal_mask=causal_mask)

        x = self.decoder_norm(x)
        return self.output_proj(x)

    def forward(
        self,
        src_tokens: torch.Tensor,
        tgt_tokens: torch.Tensor,
        graph_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full forward pass: encode obfuscated → decode clean.

        Parameters
        ----------
        src_tokens : torch.Tensor
            (batch, src_len) obfuscated source tokens.
        tgt_tokens : torch.Tensor
            (batch, tgt_len) clean target tokens (teacher-forced).
        graph_features : torch.Tensor, optional
            (batch, graph_feature_dim) spectral features.

        Returns
        -------
        torch.Tensor
            (batch, tgt_len, vocab_size) prediction logits.
        """
        encoder_output = self.encode(src_tokens, graph_features)
        return self.decode(tgt_tokens, encoder_output)

    @torch.no_grad()
    def generate(
        self,
        src_tokens: torch.Tensor,
        graph_features: Optional[torch.Tensor] = None,
        max_len: int = 512,
        temperature: float = 0.7,
        eos_token: int = 0,
    ) -> torch.Tensor:
        """
        Auto-regressively generate clean code from obfuscated input.

        Parameters
        ----------
        src_tokens : torch.Tensor
            (batch, src_len) obfuscated source tokens.
        graph_features : torch.Tensor, optional
            (batch, graph_feature_dim) spectral features.
        max_len : int
            Maximum generation length.
        temperature : float
            Sampling temperature (lower = more deterministic).
        eos_token : int
            End-of-sequence token ID.

        Returns
        -------
        torch.Tensor
            (batch, gen_len) generated token IDs.
        """
        batch_size = src_tokens.shape[0]
        device = src_tokens.device

        encoder_output = self.encode(src_tokens, graph_features)

        # Start with a BOS-like token (newline byte = 10)
        generated = torch.full(
            (batch_size, 1), fill_value=10, dtype=torch.long, device=device
        )

        for _ in range(max_len):
            logits = self.decode(generated, encoder_output)
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            # Stop if all sequences have produced EOS
            if (next_token == eos_token).all():
                break

        return generated

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
