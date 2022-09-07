import torch as t
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import wandb
import fancy_einsum as einsum
from einops import rearrange, repeat, reduce
from utils import OsSoluConfig


class OsSoluModel(nn.Module):
    def __init__(self, config: OsSoluConfig) -> None:
        super().__init__()
        normalised_shape = None             # TODO: normalised_shape should be defined properly
        self.config = config
        self.embed_positions = nn.Embedding(config.max_positional_embeddings, config.d_model)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.transformer_blocks = nn.ModuleList([GPT2Block(config) for _ in range(config.num_blocks)])
        self.final_ln = nn.LayerNorm(normalized_shape, config.ln_eps)
        self.unembed = nn

    def forward(self, x: t.Tensor) -> t.Tensor:
        positional_embeddings = self.embed_positions(t.arange(x.size(1)))
        token_embeddings = self.embed_tokens(x)
        embeddings = positional_embeddings + token_embeddings
        out = self.dropout(embeddings)
        out = self.transformer_blocks(out)

class SoLU(nn.Module):
    def __init__(self):
        pass

    def forward(self, x: t.Tensor) -> t.Tensor:
        return x * x.softmax(dim=-1)

class GPT2Block(nn.Module):
    def __init__(self, config: OsSoluConfig) -> None:
        super().__init__() 
        self.config = config

        self.layer_norm1 = nn.LayerNorm(normalized_shape, config.ln_eps)
        self.attention = UnidirectionalAttention(config) if config.self_attention_type == "unidirectional" else RotaryAttention(config)
        self.MLP = nn.Sequential(
            nn.LayerNorm(normalized_shape, config.ln_eps),
            nn.Linear(config.d_model, 4*config.d_model),
            SoLU(),
            nn.Linear(4*config.d_model, config.d_model),
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
        self.project_q = nn.Linear(config.num_embeddings, config.d_model)
        self.project_k = nn.Linear(config.num_embeddings, config.d_model)
        self.project_v = nn.Linear(config.num_embeddings, config.d_model)
        self.project_out = nn.Linear(config.d_model, config.d_model)
        self.LARGE_NEGATIVE_VALUE = -1e5

    def hidden_to_heads(self, tensor: t.Tensor) -> t.Tensor:
        return rearrange(tensor, "b s (nh hs) -> b nh s hs", nh=self.num_heads)

    def compute_pre_softmax_attn_pattern(self, x: t.Tensor) -> t.Tensor:
        Q = self.project_q(x)
        K = self.project_k(x)

        Q = self.hidden_to_heads(Q)
        K = self.hidden_to_heads(K)
        attention_pattern = einsum("batch num_heads seqlen_q head_size, batch num_heads seqlen_k head_size -> batch num_heads seqlen_q seqlen_k")

        return attention_pattern

    def forward(self, x: t.Tensor) -> t.Tensor:
        batch, seqlen, hidden_size = x.shape
        attention_pattern = self.compute_pre_softmax_attn_pattern(x)
        V = self.project_v(x)
        
        # Masking attention. Since GPT is unidirectional, it should only attend to previous tokens.
        if seqlen > 1:
            fst_range = t.arange(seqlen, device=self.device).unsqueeze(0).T
            snd_range = t.arange(seqlen, device=self.device).unsqueeze(0)
            bool_array = fst_range < snd_range
            attention_score[..., bool_array] = self.LARGE_NEGATIVE_VALUE
        
        
        attention_pattern = attention_pattern / t.sqrt(t.tensor(self.d_model // self.num_heads))
        attention_score = attention_pattern.softmax(dim=-1)
        
        V = self.hidden_to_heads(V)
        out = einsum("batch num_heads seqlen_q seqlen_k, batch num_heads seqlen_k head_size -> batch num_heads seqlen_q head_size", attention_score, V)
        out = rearrange("b nh s hs -> b s (nh hs)")
        out = self.project_out(out)
        

        return out

class RotaryAttention(nn.Module):
    def __init__(self, config: OsSoluConfig) -> None:
        super().__init__()
        self.config = config
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        # TODO: implement rotary self-attention
        pass