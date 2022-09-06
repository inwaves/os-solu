@dataclass
class OsSoluConfig:
    d_model: int = 512
    vocab_size: int = 65536 # Unsure about this.
    learning_rate: float = 1e-3
    num_embeddings: int = 1024 # Unsure about this.