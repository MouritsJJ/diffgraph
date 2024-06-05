import os
import torch
import argparse
import yaml
import importlib
from torch_geometric.loader import DataLoader

def get_model(config, device):
    """
        Setup model on device.

        Parameters
        ---
        config  : Dictionary with model name.
        devce   : CUDA device to send model to.

        Returns
        ---
        model   : DataParallel model on device.
    """
    # Get model name from dicitonary
    name = config["name"]
    # Import the model class
    model_class = getattr(importlib.import_module(f"models.{name.lower()}.model"), name)
    # Create the model and send to device
    return torch.nn.DataParallel(model_class(config["params"])).to(device)

def get_optimiser(model_parameters, config):
    """
        Setup the optimiser.

        Parameters
        ---
        model_parameters    : The parameters of the model to optimise.
        config              : Name of the optimiser and parameters for the optimiser.

        Returns
        ---
        optimiser           : The created optimiser.
    """
    # Get optimiser name
    name = config["name"]
    # Create the optimiser with given parameters
    if name == "AdamW":
        return torch.optim.AdamW(model_parameters, **config["params"])
    elif name == "Adam":
        return torch.optim.Adam(model_parameters, **config["params"])
    else:
        print('Unknown optimiser', name)
        exit()

def get_dataset(config):
    """
        Load dataset and split into training and validation sets.

        Parameters
        ---
        config  : Dictionary with entries: dataset name, batch_size, val_batch_size, train_split_size (optional)

        Returns
        ---
        train   : The dataloader for the training split.
        val     : The dataloader for the validation split.
        dataset : The dataset object.
    """
    # Get name of the dataset
    name = config["name"]
    # Import dataset class
    dataset_class = getattr(importlib.import_module(f"datasets.{name.lower()}"), name)
    # Create dataset
    dataset = dataset_class(config)

    # Split dataset into training (90%) and validation set (10%) if splitting is not done priorly
    if "train_split_size" in config and config["train_split_size"] > 0:
        train_split, val_split = dataset[:config["train_split_size"]], dataset[config["train_split_size"]:]
    else:
        train_split, val_split = torch.utils.data.random_split(dataset=dataset, lengths=[0.9, 0.1], generator=torch.Generator().manual_seed(42))
    
    # Create training and validation dataloader to iterate the dataset
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    train = DataLoader(
        train_split, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True
    )
    val = DataLoader(val_split, batch_size=config['val_batch_size'], shuffle=True)

    return train, val, dataset

def create_checkpoint(model, optim, config, epoch):
    """
        Save model and optimiser parameters as a checkpoint.

        Parameters
        ---
        model   : The model to save.
        optim   : The optimiser to save.
        config  : Dictionary with settings used to create the model and optimiser.
        epoch   : The epoch that is created a checkpoing for.
    """
    # Prepare model
    model_dict = dict(
        name=config["model"]["name"],
        params=config["model"]["params"],
        state=model.state_dict(),
    )

    # Prepare optimiser
    optim_dict = dict(
        name=config["optim"]["name"],
        params=config["optim"]["params"],
        state=optim.state_dict()
    )

    # Save model and optimiser
    torch.save(model_dict, os.path.join("runs", config["training"]["run_name"], "models", f'epoch_{epoch}_model.pt'))
    torch.save(optim_dict, os.path.join("runs", config["training"]["run_name"], "models", f'epoch_{epoch}_optim.pt'))


def load_checkpoint(device, config):
    """
        Load model and optimiser parameters from a checkpoint.

        Parameters
        ---
        device  : CUDA device to load checkpoint onto.
        config  : Dictionary with entries: resume_epoch, run_name

        Returns
        ---
        model   : The loaded model on device.
        optim   : The loaded optimiser on device.
    """
    # Get epoch to load checkpoint form
    epoch = config["training"]["resume_epoch"]
    # Load model and optimiser settings
    model_dict = torch.load(os.path.join("runs", config["training"]["run_name"], "models", f'epoch_{epoch}_model.pt'))
    optim_dict = torch.load(os.path.join("runs", config["training"]["run_name"], "models", f'epoch_{epoch}_optim.pt'))

    # Create model and optimiser
    model = get_model(model_dict, device)
    optim = get_optimiser(model.parameters(), config["optim"])

    # Load state into model and optimiser
    model.load_state_dict(model_dict["state"])
    optim.load_state_dict(optim_dict["state"])

    return model, optim

def setup_logging(save_path):
    """
        Initialises folder structure for training, evaluation and sampling.

        Parameters
        ---
        save_path   : The path to setup folders for logging.
    """
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "models"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "samples"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "eval"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "test"), exist_ok=True)

def parse_args():
    """
        Load yml config file based on path from CLI arguments.
    """
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()

    # Load config file
    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

    # Calculate batch_size based on number of GPUs
    if torch.cuda.is_available(): config["dataset"]["batch_size"] *= torch.cuda.device_count() 

    return config
