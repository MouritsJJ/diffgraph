import torch
import importlib

from utils.util import *
from utils.mol_utils import *
from utils.graph_utils import *

def sample_batch(n_samples, dataset, model, device):
    # Get categorical distribution of number of atoms in molecules in dataset
    dist_atoms = torch.distributions.categorical.Categorical(probs=dataset.node_probs)

    # Sample number of nodes in each molecule from above distribution
    nodes = dist_atoms.sample(sample_shape=(n_samples,)).to(device)
    max_nodes = torch.max(nodes).item()

    # Generate node mask
    arange = torch.arange(max_nodes, device=device).unsqueeze(0).expand(n_samples, -1)
    node_mask = arange < nodes.unsqueeze(1)
    X_mask = node_mask.unsqueeze(-1)        # bs, n, 1
    E_mask = (X_mask.transpose(1,2) * X_mask).unsqueeze(-1) # bs, n, n, 1

    # Calculate dims
    input_dim_X = model.module.input_dims['X']
    input_dim_E = model.module.input_dims['E']
    output_dim_y = model.module.input_dims['y']

    # Sample n_samples noisy graphs from a uniform distribution
    X = torch.randint(high=input_dim_X, size=(n_samples, max_nodes)).to(device)
    X = torch.nn.functional.one_hot(X, num_classes=input_dim_X)
    E = torch.randint(high=input_dim_E, size=(n_samples, max_nodes, max_nodes)).to(device)
    E = torch.nn.functional.one_hot(E, num_classes=input_dim_E)
    y = torch.zeros((n_samples, output_dim_y)).to(device)

    # Mirror edge features to ensure graph is undirected and mask unused nodes
    X = X * X_mask
    E = E * E_mask
    E = mirror(E)

    return X, E, y, node_mask

def sample_reverse(con_diffusion, cat_diffusion, model, t, x_t):
    # Unpack the data sample and compute mask
    X, E, y, node_mask = x_t
    X_mask = node_mask
    E_mask = node_mask.unsqueeze(-1).transpose(1,2) * node_mask.unsqueeze(-1)
    device = X.device

    # Prepare timestep for the batch
    t = (torch.ones(X.shape[0]) * t).long().to(device)
    y_t = torch.hstack((y[:, :-1], t.unsqueeze(-1)))

    # Reverse one step in the diffusion process
    X_0, E_0, _ = model(X.argmax(dim=-1).long(), E.argmax(dim=-1).long(), y_t, node_mask)
    X = cat_diffusion.reverse_process(X.argmax(dim=-1).long(), X_0, t, X_mask, model.module.input_dims['X'])
    E = cat_diffusion.reverse_process(E.argmax(dim=-1).long(), E_0, t, E_mask, model.module.input_dims['E'])
    E = mirror(E)

    return X, E, y, node_mask

def sample_mols(x_0, dataset):
    # Convert the graphs to RDKit molecules
    X, E, _, node_mask = x_0
    X, E = X.argmax(-1), E.argmax(-1)
    return [mol_from_graph(X_i, E_i, dataset.decode_atom, dataset.decode_bond)[0] for X_i, E_i in prepare_graph(X, E, node_mask)]
