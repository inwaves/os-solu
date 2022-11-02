import torch as t
import torch.nn as nn
from fancy_einsum import einsum
from einops import rearrange
from utils import OsSoluConfig


# TODO: Add hooks to the model.
# TODO: Add support for mixing dense and sparse attention.

class OsSoluModel(nn.Module):
    """An open-source implementation of a SoLU-based transformer. This is a GPT-style architecture model
    where the nonlinearity in the MLP block is replaced with SoLU(x) = x * softmax(x)."""

    def __init__(self, config: OsSoluConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_positions = nn.Embedding(config.max_positional_embeddings, config.d_model)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.transformer_blocks = nn.ModuleList([GPT2Block(config) for _ in range(config.num_blocks)])
        self.final_ln = nn.LayerNorm(config.d_model, config.ln_eps)

    def forward(self, x: t.Tensor) -> t.Tensor:
        positional_embeddings = self.embed_positions(t.arange(x.size(1), device=x.device))
        token_embeddings = self.embed_tokens(x)
        embeddings = positional_embeddings + token_embeddings
        out = self.dropout(embeddings)
        for block in self.transformer_blocks:
            out = block(out)

        # Unembedding is not separate, so we just einsum with token embedding weights.
        out = einsum("vocab hidden, batch seq hidden -> batch seq vocab", self.embed_tokens.weight, out)
        return out


class SoLU(nn.Module):
    """A simple wrapper around the SoLU function such that it can be used as a layer in a model."""

    def __init__(self):
        super().__init__()

    def forward(self, x: t.Tensor) -> t.Tensor:
        return x * x.softmax(dim=-1)


class GPT2Block(nn.Module):
    def __init__(self, config: OsSoluConfig) -> None:
        super().__init__()
        self.config = config

        self.layer_norm1 = nn.LayerNorm(config.d_model, config.ln_eps)
        self.attention = UnidirectionalAttention(
            config) if config.self_attention_type == "unidirectional" else RotaryAttention(config)
        nonlinearity = SoLU() if config.nonlinearity == "solu" else nn.ReLU()
        self.MLP = nn.Sequential(
                nn.LayerNorm(config.d_model, config.ln_eps),
                nn.Linear(config.d_model, 4 * config.d_model),
                nonlinearity,
                nn.Linear(4 * config.d_model, config.d_model),
                nn.Dropout(config.dropout)
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.MLP(x)
        return x


class UnidirectionalAttention(nn.Module):
    def __init__(self, config: OsSoluConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.d_model = config.d_model
        self.project_q = nn.Linear(config.d_model, config.d_model)
        self.project_k = nn.Linear(config.d_model, config.d_model)
        self.project_v = nn.Linear(config.d_model, config.d_model)
        self.project_out = nn.Linear(config.d_model, config.d_model)
        self.LARGE_NEGATIVE_VALUE = -1e5

    def hidden_to_heads(self, tensor: t.Tensor) -> t.Tensor:
        return rearrange(tensor, "b s (nh hs) -> b nh s hs", nh=self.num_heads)

    def compute_pre_softmax_attn_pattern(self, x: t.Tensor) -> t.Tensor:
        Q = self.project_q(x)
        K = self.project_k(x)

        Q = self.hidden_to_heads(Q)
        K = self.hidden_to_heads(K)
        attention_pattern = einsum(
                "batch num_heads seqlen_q head_size, "
                "batch num_heads seqlen_k head_size ->"
                "batch num_heads seqlen_q seqlen_k",
                Q, K)

        return attention_pattern

    def forward(self, x: t.Tensor) -> t.Tensor:
        batch, seqlen, hidden_size = x.shape
        attention_pattern = self.compute_pre_softmax_attn_pattern(x)
        V = self.project_v(x)

        # Masking attention. Since GPT is unidirectional, it should only attend to previous tokens.
        if seqlen > 1:
            fst_range = t.arange(seqlen, device=x.device).unsqueeze(0).T
            snd_range = t.arange(seqlen, device=x.device).unsqueeze(0)
            bool_array = fst_range < snd_range
            attention_pattern[..., bool_array] = self.LARGE_NEGATIVE_VALUE

        attention_pattern = attention_pattern / t.sqrt(t.tensor(self.d_model // self.num_heads))
        attention_score = attention_pattern.softmax(dim=-1)

        V = self.hidden_to_heads(V)
        out = einsum(
                "batch num_heads seqlen_q seqlen_k,"
                "batch num_heads seqlen_k head_size ->"
                "batch num_heads seqlen_q head_size",
                attention_score, V)

        out = rearrange(out, "b nh s hs -> b s (nh hs)")
        out = self.project_out(out)

        return out


class RotaryAttention(nn.Module):
    def __init__(self, config: OsSoluConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, x: t.Tensor) -> t.Tensor:
        # TODO: implement rotary self-attention
        pass


class LayerNorm(nn.Module):
    def __init__(self, config: OsSoluConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, x: t.Tensor) -> t.Tensor:
        # TODO: implement layernorm with hooks on normalisation only.
        pass
