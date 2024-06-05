import os
import torch
from tqdm import tqdm

from utils.graph_utils import *
from utils.mol_utils import *
from generate import sample
from metrics import *

def loss_fn(graph, con_diffusion, cat_diffusion, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Convert to X, E representation
    X, E, node_mask = to_dense(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
    X = X.to(device); E = E.to(device); y = graph.y.to(device); node_mask = node_mask.to(device)
    X_mask = node_mask
    E_mask = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-1).transpose(1,2)

    # Add noise to X, E
    t = con_diffusion.sample_timesteps(y.size(0)).to(device)
    y_t = torch.hstack((y[:,:-1], t.unsqueeze(-1)))
    X_t, noise_X = con_diffusion.forward_process(X, t, X_mask)
    E_t, noise_E = con_diffusion.forward_process(E, t, E_mask)
    E_t = mirror(E_t)
    E_t[:,torch.eye(X.shape[1]).bool()] = 0
    noise_E = mirror(noise_E)
    noise_E[:,torch.eye(X.shape[1]).bool()] = 0

    # Predict noise and calculate loss
    pred_X, pred_E, _ = model(X_t, E_t, y_t, node_mask)
    X_loss = con_diffusion.get_loss(pred_X, noise_X)
    E_loss = con_diffusion.get_loss(pred_E, noise_E)
    loss = X_loss + E_loss

    return loss, f'X: {X_loss:.4f}, E: {E_loss:.4f}'

def val_fn(val_dataset, con_diffusion, cat_diffusion, model, decode_atom, decode_bond, epoch, config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Evaluation - Sanity check
    evaluate(val_dataset, con_diffusion, model, config, decode_atom, decode_bond, epoch, device)

    # Sampling for performance validation
    samples = config['val_samples']
    mols = sample(model, con_diffusion, cat_diffusion, samples, model.module.__class__.__name__, val_dataset.dataset,
        os.path.join("runs", config["run_name"], 'samples'), f'epoch_{epoch}_', position=1
    )

    dsmiles = load_smiles(os.path.join("runs", config["run_name"], "smiles.txt"))
    valid = compute_validity(mols)[0]
    unique = compute_uniqueness(valid)
    novel = compute_novelty(unique, dsmiles)
    with open(os.path.join("runs", config["run_name"], "scores.txt"), "a") as file:
        file.write(f"{epoch}, {len(valid)/samples*100:.2f}, {len(unique)/len(valid)*100:.2f}, {len(novel)/len(unique)*100:.2f}\n")

def evaluate(val_dataset, diffusion, model, config, decode_atom, decode_bond, epoch, device):
    # Get random data point from validation set
    _, data = list(enumerate(val_dataset))[0]

    # Send graph to device
    X, E, node_mask = to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
    X = X.to(device); E = E.to(device); y = data.y.to(device); node_mask = node_mask.to(device)

    # Compute X, E msaks
    X_mask = node_mask
    E_mask = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-1).transpose(1,2)

    # Get timestep t and noisy graphs
    i = torch.randint(low=1, high=int(diffusion.timesteps * 0.5), size=(1,)).item()
    t = (torch.ones(size=(y.size(0),)) * i).long().to(device)
    X_t, _ = diffusion.forward_process(X, t, X_mask)
    E_t, _ = diffusion.forward_process(E, t, E_mask)
    E_t[:,torch.eye(X.shape[1]).bool()] = 0
    
    # Predict clean graph
    X_0, E_0 = X_t, E_t
    for j in tqdm(range(i, -1, -1), position=1, leave=False):
        t = (torch.ones(size=(y.size(0),)) * j).long().to(device)
        y_t = torch.hstack((y[:,:-1], t.unsqueeze(-1)))
        pred_X, pred_E, _ = model(X_0, E_0, y_t, node_mask)
        X_0 = diffusion.reverse_process(X_0, pred_X, t, X_mask)
        E_0 = diffusion.reverse_process(E_0, pred_E, t, E_mask)
        E_0 = mirror(E_0)
        E_0[:,torch.eye(E_0.shape[1]).bool()] = 0

    # Convert to molecules and save
    X, E = X.argmax(-1) * X_mask, E.argmax(-1) * E_mask
    X_t, E_t = X_t.argmax(-1) * X_mask, E_t.argmax(-1) * E_mask
    X_0, E_0 = X_0.argmax(-1) * X_mask, E_0.argmax(-1) * E_mask
    mols = [mol_from_graph(x_i, e_i, decode_atom, decode_bond)[0] for x_i, e_i in prepare_graph(X, E, node_mask)]
    mols_t = [mol_from_graph(x_i, e_i, decode_atom, decode_bond)[0] for x_i, e_i in prepare_graph(X_t, E_t, node_mask)]
    mols_0 = [mol_from_graph(x_i, e_i, decode_atom, decode_bond)[0] for x_i, e_i in prepare_graph(X_0, E_0, node_mask)]
    save_molecules(mols, os.path.join("runs", config["run_name"], 'eval', f'{epoch}_{i}_val_orig.jpg'))
    save_molecules(mols_t, os.path.join("runs", config["run_name"], 'eval', f'{epoch}_{i}_val_noisy.jpg'))
    save_molecules(mols_0, os.path.join("runs", config["run_name"], 'eval', f'{epoch}_{i}_val_clean.jpg'))
