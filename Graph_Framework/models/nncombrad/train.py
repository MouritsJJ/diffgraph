import os
import torch
import torch_geometric as pyg
from tqdm import tqdm

from utils.graph_utils import *
from utils.mol_utils import *
from generate import sample
from metrics import *

def loss_fn(graph, con_diffusion, cat_diffusion, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Convert to X, E, A, L, T representation
    X, E, node_mask = to_dense(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
    A, _ = pyg.utils.to_dense_batch(x=graph.angles, batch=graph.batch)
    L, _ = pyg.utils.to_dense_batch(x=graph.langles, batch=graph.batch)
    T, _ = pyg.utils.to_dense_batch(x=graph.tangles, batch=graph.batch)
    X, E, y = X.long().to(device), E.long().to(device), graph.y.to(device)
    A, L, T = A.to(device), L.to(device), T.to(device)

    # X, E masks
    node_mask = node_mask.to(device)
    X_mask = node_mask # B, N
    E_mask = node_mask.unsqueeze(-1).transpose(1,2) * node_mask.unsqueeze(-1) # B, N, N

    # Add noise to atom and bonds
    t = cat_diffusion.sample_timesteps(y.size(0)).to(device)
    y_t = torch.hstack((y[:,:-1], t.unsqueeze(-1)))
    X, X_t = cat_diffusion.forward_process(X, t, X_mask, model.module.input_dims['X'])
    E, E_t = cat_diffusion.forward_process(E, t, E_mask, model.module.input_dims['E'])
    E_t = mirror(E_t)

    # Add noise to angles
    A_mask = get_triplet_mask(E_t.argmax(-1).long(), model.module.input_dims['A'])
    L_mask = get_line_mask(E_t.argmax(-1).long(), model.module.input_dims['L'])
    T_mask = get_twist_mask(E_t.argmax(-1).long(), model.module.input_dims['T'])
    A_t, noise_A = con_diffusion.forward_process(A, t, A_mask)
    L_t, noise_L = con_diffusion.forward_process(L, t, L_mask)
    T_t, noise_T = con_diffusion.forward_process(T, t, T_mask)

    # Predict noise and calculate loss
    pred_X, pred_E, pred_A, pred_L, pred_T, _ = model(X_t.argmax(dim=-1).long(), E_t.argmax(dim=-1).long(), A_t, L_t, T_t, y_t, node_mask)
    X_loss = cat_diffusion.get_loss(X, X_t, pred_X, t, X_mask, model.module.input_dims['X'])
    E_loss = cat_diffusion.get_loss(E, E_t, pred_E, t, E_mask, model.module.input_dims['E'])
    A_loss = con_diffusion.get_loss(pred_A * A_mask, noise_A * A_mask)
    L_loss = con_diffusion.get_loss(pred_L * L_mask, noise_L * L_mask)
    T_loss = con_diffusion.get_loss(pred_T * T_mask, noise_T * T_mask)
    loss = X_loss + E_loss + A_loss + L_loss + T_loss

    return loss, f'X: {X_loss:.4f}, E: {E_loss:.4f}, A: {A_loss:.4f}, L: {L_loss:.4f}, T: {T_loss:.4f}'

def val_fn(val_dataset, con_diffusion, cat_diffusion, model, decode_atom, decode_bond, epoch, config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Evaluation - Sanity check
    evaluate(val_dataset, con_diffusion, cat_diffusion, model, config, decode_atom, decode_bond, epoch, device)

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

def evaluate(val_dataset, con_diffusion, cat_diffusion, model, config, decode_atom, decode_bond, epoch, device):
    # Get random data point from validation set
    _, graph = list(enumerate(val_dataset))[0]

    # Convert to X, E, A, L, T representation
    X, E, node_mask = to_dense(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
    A, _ = pyg.utils.to_dense_batch(x=graph.angles, batch=graph.batch)
    L, _ = pyg.utils.to_dense_batch(x=graph.langles, batch=graph.batch)
    T, _ = pyg.utils.to_dense_batch(x=graph.tangles, batch=graph.batch)
    X, E, y = X.long().to(device), E.long().to(device), graph.y.to(device)
    A, L, T = A.to(device), L.to(device), T.to(device)

    # X, E masks
    node_mask = node_mask.to(device)
    X_mask = node_mask # B, N
    E_mask = node_mask.unsqueeze(-1).transpose(1,2) * node_mask.unsqueeze(-1) # B, N, N

    # Get timestep t and noisy graphs
    i = torch.randint(low=1, high=int(cat_diffusion.timesteps * 0.5), size=(1,)).item()
    t = (torch.ones(size=(y.size(0),)) * i).long().to(device)
    X, X_t = cat_diffusion.forward_process(X, t, X_mask, model.module.input_dims['X'])
    E, E_t = cat_diffusion.forward_process(E, t, E_mask, model.module.input_dims['E'])
    E_t = mirror(E_t)

    # Add noise to angles
    A_mask = get_triplet_mask(E_t.argmax(-1).long(), model.module.input_dims['A'])
    L_mask = get_line_mask(E_t.argmax(-1).long(), model.module.input_dims['L'])
    T_mask = get_twist_mask(E_t.argmax(-1).long(), model.module.input_dims['T'])
    A_t, _ = con_diffusion.forward_process(A, t, A_mask)
    L_t, _ = con_diffusion.forward_process(L, t, L_mask)
    T_t, _ = con_diffusion.forward_process(T, t, T_mask)

    # Predict clean graph
    X_0, E_0, A_0, L_0, T_0 = X_t, E_t, A_t, L_t, T_t
    for j in tqdm(range(i, -1, -1), position=1, leave=False):
        t = (torch.ones(size=(y.size(0),)) * j).long().to(device)
        y_t = torch.hstack((y[:,:-1], t.unsqueeze(-1)))
        pred_X_0, pred_E_0, pred_A_0, pred_L_0, pred_T_0, _ = model(X_0.argmax(dim=-1).long(), E_0.argmax(dim=-1).long(), A_0, L_0, T_0, y_t, node_mask)
        X_0 = cat_diffusion.reverse_process(X_0.argmax(dim=-1).long(), pred_X_0, t, X_mask, model.module.input_dims['X'])
        E_0 = cat_diffusion.reverse_process(E_0.argmax(dim=-1).long(), pred_E_0, t, E_mask, model.module.input_dims['E'])
        E_0 = mirror(E_0)

        A_mask = get_triplet_mask(E_0.argmax(-1).long(), model.module.input_dims['A'])
        L_mask = get_line_mask(E_0.argmax(-1).long(), model.module.input_dims['L'])
        T_mask = get_twist_mask(E_0.argmax(-1).long(), model.module.input_dims['T'])
        A_0 = con_diffusion.reverse_process(A_0, pred_A_0, t, A_mask)
        L_0 = con_diffusion.reverse_process(L_0, pred_L_0, t, L_mask)
        T_0 = con_diffusion.reverse_process(T_0, pred_T_0, t, T_mask)

    # Convert to molecules and save
    X, E = X.argmax(-1) * X_mask, E.argmax(-1) * E_mask
    X_0, E_0 = X_0.argmax(-1) * X_mask, E_0.argmax(-1) * E_mask
    X_t, E_t = X_t.argmax(-1) * X_mask, E_t.argmax(-1) * E_mask
    mols = [mol_from_graph(x_i, e_i, decode_atom, decode_bond)[0] for x_i, e_i in prepare_graph(X, E, node_mask)]
    mols_0 = [mol_from_graph(x_i, e_i, decode_atom, decode_bond)[0] for x_i, e_i in prepare_graph(X_0, E_0, node_mask)]
    mols_t = [mol_from_graph(x_i, e_i, decode_atom, decode_bond)[0] for x_i, e_i in prepare_graph(X_t, E_t, node_mask)]
    save_molecules(mols, os.path.join("runs", config["run_name"], 'eval', f'{epoch}_{i}_val_orig.jpg'))
    save_molecules(mols_t, os.path.join("runs", config["run_name"], 'eval', f'{epoch}_{i}_val_noisy.jpg'))
    save_molecules(mols_0, os.path.join("runs", config["run_name"], 'eval', f'{epoch}_{i}_val_clean.jpg'))
    