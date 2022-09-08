import numpy as np
from einops import rearrange

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

def tokenise(batch, tokeniser, num_gpus: int = 1, context_length: int = 1024):
    """Tokenise a batch of text data. This implementation is idiosyncratic to the Pile dataset, but can be easily modified to work with e.g. C4.

    Args:
        batch (dict): The batch of text, as a dict with a 'text' field.
        tokeniser (-): A huggingface-API tokeniser, of type returned by AutoTokenizer.from_pretrained (depends on model chosen).
        num_gpus (int, optional): The number of GPUs available for data parallel training. Defaults to 1.
        context_length (int, optional): The context length of the model that will be trained on this data. Defaults to 1024.

    Returns:
        dict: A single field dictionary, 'text', whose value is a tensor of shape (batch_size, sequence_length) containing tokenised sequences.
    """
    batch = batch["text"]
    full_text = tokeniser.eos_token.join(batch)

    # Divide entire batch among all GPUs available.
    seq_len = len(full_text)//num_gpus
    sequence_list = [full_text[i*seq_len:(i+1)*seq_len] for i in range(num_gpus)]

    # Tokenise sequences, removing padding tokens.
    all_tokens = tokeniser(sequence_list, return_tensors="pt", padding=True)["input_ids"].flatten()
    all_tokens = all_tokens[all_tokens != tokeniser.pad_token_id]

    # Reshape all_tokens to be (batch_size x sequence_length) where each sequence has
    # a "beginning of sequence" token prepended to it.
    num_tokens = len(all_tokens)
    current_batch_size = num_tokens // (context_length-1)
    all_tokens = all_tokens[:(context_length-1)*current_batch_size]
    all_tokens = rearrange(all_tokens, "(batch_size seq_len) -> batch_size seq_len", batch_size=current_batch_size, seq_len=context_length-1)
    prefix = np.full((current_batch_size, 1), tokeniser.bos_token_id, dtype=np.int64)

    tokenised_text = np.concatenate([prefix, all_tokens], axis=1)
    assert  tokenised_text.shape == (current_batch_size, context_length)
    print(f"{current_batch_size=}, {context_length=}")
    return {"text": tokenised_text}

    