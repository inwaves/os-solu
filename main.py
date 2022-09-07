import torch as t
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import argparse
from utils import OsSoluConfig
from model import OsSoluModel
from typing import Tuple

def parse_arguments() -> argparse.Namespace:
    # TODO: command-line args for hparams
    parser = argparse.ArgumentParser(description="Parse command-line arguments for this model.")
    parser.add_argument("--d_model", type=int, default=512, help="Hidden size of the model.")
    parser.add_argument("--vocab_size", type=int, default=65536, help="Vocabulary size of the input sequence.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimiser.")
    parser.add_argument("--num_embeddings", type=int, default=1024, help="Number of embeddings.")
    parser.add_argument("--num_blocks", type=int, default=1, help="Number of transformer blocks.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Probability of dropout.")
    parser.add_argument("--ln_eps", type=float, default=1e-3, help="Layer norm epsilon.")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads in each attention layer.")
    parser.add_argument("--self_attention_type", type=str, default="unidirectional", help="What type of attention to use: rotary or unidirectional. ")
    parser.add_argument("--max_positional_embeddings", type=int, default=1024, help="Maximum number of positional embeddings.")
    args = parser.parse_args()
    return args

def train(config: OsSoluConfig, model: OsSoluModel) -> OsSoluModel:
    # TODO: training loop
    
    return model

def eval():
    pass

def setup() -> Tuple[OsSoluConfig, OsSoluModel]:
    # TODO: wandb logging
    args = parse_arguments()
    config = OsSoluConfig(args)
    model = OsSoluModel(config)
    return config, model

if __name__=="__main__":
    config, model = setup()
    trained_model = train(config, model)
    eval()