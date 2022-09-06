import torch as t
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import wandb
import fancy_einsum
from einops import rearrange, repeat, reduce


class OsSoluModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.transformer_block = TransformerBlock(config)

    def forward(self, x: t.Tensor) -> t.Tensor:
        pass


class TransformerBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.num_embeddings, config.d_model)
        self.linear = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            SoLU(),
        )
        self.layer_norm = nn.LayerNorm(normalized_shape)
        self.unembed = nn.Embedding(config.num_embeddings, config.d_model)

    def forward(self, x: t.Tensor) -> t.Tensor:
        pass


class RotaryAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

    def forward(self, x: t.Tensor, attention_mask: t.Tensor) -> t.Tensor:
        # Compute pre-softmax attention scores
        # Apply attention mask
        # Compute softmax
        # Apply final einsum
        # Return attention output

        pass
