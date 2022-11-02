import argparse
import time
import torch as t
import torch.optim as optim
from tqdm import tqdm
import wandb

from typing import Tuple
from torch.utils.data.dataloader import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from utils import OsSoluConfig, tokenise, loss_fn, count_parameters
from model import OsSoluModel

WANDB_PROJECT_NAME = "os_solu"
DEVICE = "cuda" if t.cuda.is_available() else "cpu"


# TODO: Add support for distributed training.
# TODO: Use only book data from dataset.

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
    parser.add_argument("--batch_size", type=int, default=40, help="Batch size used in training.")
    parser.add_argument("--checkpoint_every_n_tokens", type=int, default=500_000_000,
                        help="Save a checkpoint of the model every n tokens processed.")
    parser.add_argument("--d_model", type=int, default=512, help="Hidden size of the model.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Probability of dropout.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimiser.")
    parser.add_argument("--ln_eps", type=float, default=1e-3, help="Layer norm epsilon.")
    parser.add_argument("--max_positional_embeddings", type=int, default=1024,
                        help="Maximum number of positional embeddings/sequence length.")
    parser.add_argument("--nonlinearity", type=str, default="solu",
                        help=" Nonlinearity to use inside MLP block: must be relu or solu.")
    parser.add_argument("--num_blocks", type=int, default=1, help="Number of transformer blocks.")
    parser.add_argument("--num_embeddings", type=int, default=1024, help="Number of embeddings.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to run for.")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads in each attention layer.")
    parser.add_argument("--optimiser_type", type=str, default="adam", help="Optimiser type.")
    parser.add_argument("--self_attention_type", type=str, default="unidirectional",
                        help="What type of attention to use: rotary or unidirectional.")
    parser.add_argument("--vocab_size", type=int, default=50_278, help="Vocabulary size of the input sequence.")
    args = vars(parser.parse_args())

    # Parse string arguments.
    allowed_values = {
            "optimiser_type":      ["adam", "sgd"],
            "self_attention_type": ["unidirectional", "rotary"],
            "nonlinearity":        ["relu", "solu"],
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
    wandb.watch(model, criterion=loss_fn, log="all", log_freq=10, log_graph=True)

    # Initialise optimiser.
    opt = optim.Adam if config.optimiser_type.lower() == "adam" else optim.SGD
    optimiser = opt(model.parameters(), lr=config.learning_rate)

    # Train loop.
    examples_seen = 0
    train_data_iterator = iter(train_dataloader)
    for epoch in range(config.num_epochs):
        for i, batch in enumerate(tqdm(train_data_iterator
                                       )):
            start_time = time.time()
            batch = batch["text"]
            batch = batch.to(DEVICE)

            logits = model(batch)
            optimiser.zero_grad()
            loss = loss_fn(logits, batch)
            loss.backward()
            optimiser.step()

            wandb.log(dict(train_loss=loss, elapsed=time.time() - start_time), step=examples_seen)

            # Number of tokens processed is batch_size * sequence_length.
            examples_seen += batch.numel()

            # Save a checkpoint of the model.
            if examples_seen % config.checkpoint_every_n_tokens == 0:
                # Save the model's state on disk, then upload to wandb.
                filename = f"{wandb.run.dir}/os_solu_model_ckpt_step_{examples_seen}.pt"
                t.save({
                        "step":                 examples_seen,
                        "model_state_dict":     model.state_dict(),
                        "optimiser_state_dict": optimiser.state_dict(),
                        "loss":                 loss.item()
                }, filename)
                wandb.save(filename)
                print(f"Checkpointing model at {examples_seen} tokens seen.")

    return model


def evaluate(model: OsSoluModel, test_dataloader: DataLoader) -> None:
    """Evaluates a trained model on the test dataset provided.

    Args:
        model (OsSoluModel): The trained model.
        test_dataloader (t.utils.data.DataLoader): The dataset on which to evaluate the model.
    """
    # Eval loop.
    examples_seen = 0
    total_loss, num_correct = 0, 0
    model.eval()
    with t.inference_mode():
        test_data_iterator = iter(test_dataloader)
        start_time = time.time()
        for i, batch in enumerate(tqdm(test_data_iterator)):
            batch = batch["text"]
            batch = batch.to(DEVICE)

            logits = model(batch)
            total_loss += loss_fn(logits, batch).item()
            examples_seen += len(batch)
        wandb.log(dict(test_loss=total_loss, elapsed=time.time() - start_time), step=examples_seen)

    # Save the model's state on disk, then upload to wandb.
    filename = f"{wandb.run.dir}/model_state_dict.pt"
    t.save(model.state_dict(), filename)
    wandb.save(filename)


def setup() -> Tuple[OsSoluConfig, OsSoluModel, DataLoader, DataLoader]:
    """This function delegates the setup to various helper functions.

    Returns:
        Tuple[OsSoluConfig, OsSoluModel, t.utils.data.DataLoader, t.utils.data.DataLoader]:
        A tuple containing a config, a model, a training dataset and a test dataset.
    """
    args = parse_arguments()
    config = OsSoluConfig(args)
    model = OsSoluModel(config).to(DEVICE)
    args["num_parameters"] = count_parameters(model)
    wandb.init(project=WANDB_PROJECT_NAME, config=args)

    start_data_time = time.time()
    # Load and prep data.
    ds = load_dataset("the_pile", streaming=True)

    try:
        ds = ds.remove_columns("meta")
    except:
        print("Dataset did not contain 'meta' column.")

    train_dataset = ds["train"]
    test_dataset = ds["test"]

    tokeniser = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokeniser.add_special_tokens({"pad_token": "<PAD>"})

    train_dataset = train_dataset.map(lambda x: tokenise(x, tokeniser, 1, config.max_positional_embeddings),
                                      batched=True).with_format("torch")
    test_dataset = test_dataset.map(tokenise, batched=True).with_format("torch")

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size)
    print(f"Data loaded in {time.time() - start_data_time:.1f}s.")

    return config, model, train_dataloader, test_dataloader


if __name__ == "__main__":
    config, model, train_dataloader, test_dataloader = setup()
    trained_model = train(config, model, train_dataloader)
    evaluate(trained_model, test_dataloader)
