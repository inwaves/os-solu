@dataclass
class OsSoluConfig:
    d_model: int = 512                  # Hidden size of the model.
    vocab_size: int = 65536             # Vocabulary size of the input sequence. Unsure about this.
    learning_rate: float = 1e-3         # Learning rate for the optimiser.
    num_embeddings: int = 1024          # Number of embeddings. Unsure about this.
    num_blocks: int = 1                 # Number of transformer blocks.
    dropout: float = 0.1                # Probability of dropout.
    ln_eps: float = 1e-3                # Layer norm epsilon.
    num_heads: int = 4                  # Number of attention heads in each attention layer.