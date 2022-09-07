import argparse
import torch as t
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from tqdm import tqdm
import wandb

from typing import Tuple
from torch.utils.data.dataloader import DataLoader
from datasets import load_dataset
from utils import OsSoluConfig
from model import OsSoluModel

WANDB_PROJECT_NAME = "os_solu"
DEVICE = "cuda" if t.cuda.is_available() else "cpu"

def parse_arguments() -> dict:
    """Parses command-line arguments for this model run. Arguments of type string have allowed values, 
       which are enforced. Default parameter values are provided such that fields in the config are never None.

    Raises:
        ValueError: optimiser type must be adam or sgd.
        ValueError: attention type must be rotary or unidirectional.

    Returns:
        dict: a dictionary containing the command-line arguments parsed by this function.
    """
    parser = argparse.ArgumentParser(description="Parse command-line arguments for this model.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size used in training.")
    parser.add_argument("--d_model", type=int, default=512, help="Hidden size of the model.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Probability of dropout.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimiser.")
    parser.add_argument("--ln_eps", type=float, default=1e-3, help="Layer norm epsilon.")
    parser.add_argument("--max_positional_embeddings", type=int, default=1024, help="Maximum number of positional embeddings.")
    parser.add_argument("--nonlinearity", type=str, default="solu", help=" Nonlinearity to use inside MLP block: must be relu or solu.")
    parser.add_argument("--num_blocks", type=int, default=1, help="Number of transformer blocks.")
    parser.add_argument("--num_embeddings", type=int, default=1024, help="Number of embeddings.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs to run for.")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads in each attention layer.")
    parser.add_argument("--optimiser_type", type=str, default="adam", help="Optimiser type.")
    parser.add_argument("--self_attention_type", type=str, default="unidirectional", help="What type of attention to use: rotary or unidirectional.")
    parser.add_argument("--vocab_size", type=int, default=65536, help="Vocabulary size of the input sequence.")
    args = vars(parser.parse_args())

    # Parse string arguments.
    allowed_values = {
        "optimiser_type": ["adam", "sgd"], 
        "self_attention_type": ["unidirectional", "rotary"], 
        "nonlinearity": ["relu", "solu"],    
    }

    for key, values in allowed_values.items():
        if args[key] not in values:
            raise ValueError(f"{key} should be one of {values}.")

    return args

def train(config: OsSoluConfig, model: OsSoluModel, train_dataloader: DataLoader) -> OsSoluModel:
    """Trains a model using the config and training dataset provided.

    Args:
        config (OsSoluConfig): The config object.
        model (OsSoluModel): The model to train.
        train_dataloader (t.utils.data.DataLoader): The training dataset provided as a torch DataLoader object.

    Returns:
        OsSoluModel: The trained model.
    """
    # TODO: training loop
    train_loss_fn = t.nn.CrossEntropyLoss()
    wandb.watch(model, criterion=train_loss_fn, log="all", log_freq=10, log_graph=True)

    # Initialise optimiser.
    opt = optim.Adam if config.optimiser_type.lower() == "adam" else optim.SGD
    optimiser = opt(model.parameters(), lr=config.learning_rate)

    # Train loop.
    examples_seen = 0
    for epoch in range(config.num_epochs):
        for i, (data, target) in enumerate(tqdm(train_dataloader)):
            print(data, target)
            data = data.to(DEVICE)
            target = target.to(DEVICE)

            predictions = model(data)
            accuracy = (predictions.argmax(dim=-1) == target).sum() / len(data)
            optimiser.zero_grad()
            loss = train_loss_fn(target, predictions)
            loss.backward()
            optimiser.step()

            wandb.log(dict(train_loss=loss, train_accuracy=accuracy, elapsed=time.time() - start_time), step=examples_seen)
            examples_seen += len(data)

    return model

def eval(model: OsSoluModel, test_dataloader: DataLoader) -> None:
    """Evaluates a trained model on the test dataset provided.

    Args:
        model (OsSoluModel): The trained model.
        test_dataset (t.utils.data.Dataset): The dataset on which to evaluate the model.
    """
    test_loss_fn = t.nn.CrossEntropyLoss()

    # Eval loop.
    examples_seen = 0
    total_loss, num_correct = 0, 0
    model.eval()
    with t.inference_mode():
        for i, (data, target) in enumerate(tqdm(test_dataloader)):
            data = data.to(DEVICE)
            target = target.to(DEVICE)

            predictions = model(data)
            num_correct += (predictions.argmax(dim=-1) == target).sum().item()
            total_loss += test_loss_fn(target, predictions).item()
            examples_seen += len(data)
        wandb.log(dict(test_loss=total_loss, test_accuracy=num_correct / examples_seen, elapsed=time.time() - start_time), step=examples_seen)
    
    # Save the model's state on disk, then upload to wandb.
    filename = f"{wandb.run.dir}/model_state_dict.pt"
    t.save(model.state_dict(), filename)
    wandb.save(filename)


def setup() -> Tuple[OsSoluConfig, OsSoluModel]:
    """This function delegates the setup to various helper functions.

    Returns:
        Tuple[OsSoluConfig, OsSoluModel, datasets.iterable_dataset.IterableDataset, datasets.iterable_dataset.IterableDataset]: A tuple containing a config, a model, a training dataset and a test dataset.
    """
    args = parse_arguments()
    wandb.init(project=WANDB_PROJECT_NAME, config=args)
    config = OsSoluConfig(args)
    model = OsSoluModel(config)

    # Load and prep data.
    ds = load_dataset("the_pile", streaming=True)
    train_dataset = ds["train"].with_format("torch")
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size)

    test_dataset = ds["test"].with_format("torch")
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size)
    return config, model, (train_dataloader, test_dataloader)

if __name__=="__main__":
    config, model, (train_dataloader, test_dataloader) = setup()
    trained_model = train(config, model, train_dataloader)
    eval(trained_model, test_dataloader)