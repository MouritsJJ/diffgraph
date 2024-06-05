import torch
import os
import importlib
from datetime import datetime
from tqdm import tqdm
from rdkit import Chem

from utils.util import *
from metrics import *
from diffusion.con_diffusion import ContinuousDiffusion
from diffusion.cat_diffusion import CategoricalDiffusion

def sample(model, con_diffusion, cat_diffusion, n_samples, model_name, dataset_config, save_path:str, prefix='', position=0):
    """
        Sample molecules using supplied model.

        Parameters
        ---
        model           : The denoiser model.
        con_diffusion   : Continuous diffusion process.
        cat_diffusion   : Categorical diffusion process.
        n_samples       : Number of samples to sample.
        model_name      : Model name for importing sampling methods.
        dataset_config  : Dataset settings used for sampling.
        save_path       : Path to save images of examples of the sampled molecules.
        prefix          : File prefix for the file name of the saved images.
        position        : tqdm position for progress bar during sampling.

        Returns
        ---
        mols            : Sampled molecules.
    """
    # Get device and set model to evaluation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    
    # Import sampling methods using the model name to get the file path
    sample_batch = getattr(importlib.import_module(f"models.{model_name.lower()}.sample"), "sample_batch")
    sample_reverse = getattr(importlib.import_module(f"models.{model_name.lower()}.sample"), "sample_reverse")
    sample_mols = getattr(importlib.import_module(f"models.{model_name.lower()}.sample"), "sample_mols")

    # Get dataset settings for sampling
    if isinstance(dataset_config, dict):
        dataset_class = getattr(importlib.import_module(f"datasets.{dataset_config['name'].lower()}"), dataset_config['name'])
        dataset = dataset_class(dataset_config)
    else: dataset = dataset_config

    # Sample a random batch the denoise
    data_tuple = sample_batch(n_samples, dataset, model, device)
    
    with torch.no_grad():
        # Reverse the noisy batch for T timesteps
        for i in tqdm(range(con_diffusion.timesteps - 1, -1, -1), position=position, leave=position==0):
            data_tuple = sample_reverse(con_diffusion, cat_diffusion, model, i, data_tuple)

        # Convert graphs to molecules
        mols = sample_mols(data_tuple, dataset)
        # Save the molecules
        save_molecules(mols[:20], os.path.join(save_path, f"{prefix}mol.jpg"))

    # Set the model to training mode
    model.train()
    
    return mols

def main():
    mols = []
    # Parse arguments
    config = parse_args()

    # Create save folder
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    save_path = os.path.join("runs", config["sampling"]["run_name"], "test", date_time)
    os.makedirs(save_path, exist_ok=True)

    # Setup model and diffusion processes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = torch.load(config["sampling"]["model_path"], map_location=device)
    model = get_model(model_dict, device)
    model.load_state_dict(model_dict["state"])
    con_diffusion = ContinuousDiffusion(config["diffusion"], device).to(device)
    cat_diffusion = CategoricalDiffusion(config["diffusion"], device).to(device)

    # Samples molecules in batches
    l = config["sampling"]["n_samples"] // config["dataset"]["batch_size"] + 1
    print(f"Sampling {config['sampling']['n_samples']} new molecules...")
    for i in tqdm(range(1, l + 1), position=0):
        samples = sample(
            model, 
            con_diffusion,
            cat_diffusion, 
            min(config["dataset"]["batch_size"], config["sampling"]["n_samples"] - (i - 1) * config["dataset"]["batch_size"]),
            model_dict["name"],
            config["dataset"],
            save_path, 
            prefix=str(i) + '_', 
            position=1
        )
        mols.extend(samples)

    with open(os.path.join("runs", config["sampling"]["run_name"], 'test', f'{date_time}_data.sdf'), 'w') as fw:
        for mol in mols:
            conf = mol.GetConformers()[0].GetId() if len(mol.GetConformers()) > 0 else -1
            txt = Chem.SDWriter.GetText(mol, confId=conf, kekulize=False)
            fw.write(txt)

if __name__ == '__main__':
    main()
