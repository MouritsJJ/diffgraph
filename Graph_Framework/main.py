import os
import torch
import importlib
from tqdm import tqdm

from utils.util import *
from metrics import compute_dataset_smiles
from diffusion.con_diffusion import ContinuousDiffusion
from diffusion.cat_diffusion import CategoricalDiffusion

def train(config):
    # Setup folder for checkpoints and training device
    setup_logging(os.path.join("runs", config["training"]["run_name"]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataloader, decoders and diffusion
    train, val, dataset = get_dataset(config["dataset"])
    con_diffusion = ContinuousDiffusion(config["diffusion"], device).to(device)
    cat_diffusion = CategoricalDiffusion(config["diffusion"], device).to(device)

    # Create model and optimiser
    if config["training"]["resume_epoch"] > 0:
        print(f'Resuming from epoch {config["training"]["resume_epoch"]}')
        model, optimiser = load_checkpoint(device, config)
    else:
        model = get_model(config["model"], device)
        optimiser = get_optimiser(model.parameters(), config["optim"])

    loss_fn = getattr(importlib.import_module(f"models.{config['model']['name'].lower()}.train"), "loss_fn")
    val_fn = getattr(importlib.import_module(f"models.{config['model']['name'].lower()}.train"), "val_fn")

    # Compute smiles for the dataset to compute metrics when sampling
    compute_dataset_smiles(train, dataset.decode_atom, dataset.decode_bond, os.path.join("runs", config["training"]["run_name"], 'smiles.txt'))
    compute_dataset_smiles(val, dataset.decode_atom, dataset.decode_bond, os.path.join("runs", config["training"]["run_name"], 'smiles_test.txt'))
    
    # Training - Iterate every epoch
    print("Starting training")
    epoch_loss = []
    ebar = tqdm(range(config["training"]["resume_epoch"] + 1, config["training"]["epochs"] + 1), position=0, leave=True)
    for epoch in ebar:
        # Iteratate every batch
        pbar = tqdm(train, position=1, leave=False)
        for x in pbar:
            # Reset gradients, calculate loss, calculate new gradients, take gradient descent step
            optimiser.zero_grad()
            loss, log = loss_fn(x, con_diffusion, cat_diffusion, model)
            epoch_loss.append(loss.item())
            loss.backward()
            optimiser.step()

            # Update logging
            pbar.set_postfix(loss=loss.item(), log=log)

        # Run validation
        if epoch % config["training"]["val_n_epochs"] == 0: 
            create_checkpoint(model, optimiser, config, epoch)
            model.eval()
            with torch.no_grad():
                val_fn(val, con_diffusion, cat_diffusion, model, dataset.decode_atom, dataset.decode_bond, epoch, config["training"])
            model.train()
        
        with open(os.path.join("runs", config["training"]["run_name"], "loss.txt"), "a") as file:
            file.write(f'{epoch:<5n}: {torch.Tensor(epoch_loss).mean().item():.2f}\n')
            epoch_loss = []

def main():
    train(parse_args())

if __name__ == '__main__':
    main()
