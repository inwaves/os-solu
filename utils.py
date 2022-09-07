class OsSoluConfig:
    """A class to hold hyperparameters for the model itself and for the training process."""
    
    batch_size: int                             # Training data batch size.
    d_model: int                                # Hidden size of the model.
    dropout: float                              # Probability of dropout.
    learning_rate: float                        # Learning rate for the optimiser.
    ln_eps: float                               # Layer norm epsilon.
    max_positional_embeddings: int              # Maximum number of positional embeddings.
    nonlinearity: str                           # Nonlinearity to use inside MLP block: must be ReLU or SoLU.
    num_blocks: int                             # Number of transformer blocks.
    num_embeddings: int                         # Number of embeddings. Unsure about this.
    num_epochs: int                             # Number of epochs for this run.
    num_heads: int                              # Number of attention heads in each attention layer.
    self_attention_type: str                    # What type of attention to use: rotary or unidirectional. 
    optimiser_type: str                         # Optimiser type: SGD, Adam.
    vocab_size: int                             # Vocabulary size of the input sequence. Unsure about this.

    def __init__(self, args: dict) -> None:
        """Initialise this config class with values provided by a command-line argument parser.
           Values are never None here, as we provide suitable defaults in the parser call."""
        self.batch_size = args["batch_size"]
        self.d_model = args["d_model"]
        self.dropout = args["dropout"]
        self.learning_rate = args["learning_rate"]
        self.ln_eps = args["ln_eps"]
        self.max_positional_embeddings = args["max_positional_embeddings"]
        self.nonlinearity = args["nonlinearity"]
        self.num_blocks = args["num_blocks"]
        self.num_embeddings = args["num_embeddings"]
        self.num_epochs = args["num_epochs"]
        self.num_heads = args["num_heads"]
        self.optimiser_type = args["optimiser_type"]
        self.self_attention_type = args["self_attention_type"]
        self.vocab_size = args["vocab_size"]