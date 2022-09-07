import argparse

class OsSoluConfig:
    d_model: int                                # Hidden size of the model.
    vocab_size: int                             # Vocabulary size of the input sequence. Unsure about this.
    learning_rate: float                        # Learning rate for the optimiser.
    num_embeddings: int                         # Number of embeddings. Unsure about this.
    num_blocks: int                             # Number of transformer blocks.
    dropout: float                              # Probability of dropout.
    ln_eps: float                               # Layer norm epsilon.
    num_heads: int                              # Number of attention heads in each attention layer.
    self_attention_type: str                    # What type of attention to use: rotary or unidirectional. 
    max_positional_embeddings: int              # Maximum number of positional embeddings.
    
    def __init__(self, args: argparse.Namespace) -> None:
        """Initialise this config class with values provided by a command-line argument parser.
           Values are never None here, as we provide suitable defaults in the parser call."""
        self.d_model = args.d_model
        self.vocab_size = args.vocab_size
        self.learning_rate = args.learning_rate
        self.num_embeddings = args.num_embeddings
        self.num_blocks = args.num_blocks
        self.dropout = args.dropout
        self.ln_eps = args.ln_eps
        self.num_heads = args.num_heads
        self.self_attention_type = args.self_attention_type
        self.max_positional_embeddings = args.max_positional_embeddings