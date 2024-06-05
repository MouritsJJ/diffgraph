import torch
from rdkit.Chem.rdchem import Conformer
from rdkit.Geometry import Point3D

from utils.util import *
from utils.mol_utils import *
from utils.graph_utils import *

def sample_batch(n_samples, dataset, model, device):
    # Get categorical distribution of number of atoms in molecules in dataset
    dist_atoms = torch.distributions.categorical.Categorical(probs=dataset.node_probs)

    # Sample number of nodes in each molecule from distribution above
    nodes = dist_atoms.sample(sample_shape=(n_samples,)).to(device)
    max_nodes = torch.max(nodes).item()

    # Generate node mask
    arange = torch.arange(max_nodes, device=device).unsqueeze(0).expand(n_samples, -1)
    node_mask = arange < nodes.unsqueeze(1)
    X_mask = node_mask.unsqueeze(-1) # bs, n, 1
    E_mask = (X_mask.transpose(1,2) * X_mask).unsqueeze(-1) # bs, n, n, 1

    # Calculate dims
    input_dim_X  = model.module.input_dims['X']
    input_dim_E  = model.module.input_dims['E']
    input_dim_y  = model.module.input_dims['y']
    input_size_A = model.module.input_sizes['A']
    input_dim_A  = model.module.input_dims['A']
    input_size_L = model.module.input_sizes['L']
    input_dim_L  = model.module.input_dims['L']
    input_size_T = model.module.input_sizes['T']
    input_dim_T  = model.module.input_dims['T']

    # Sample n_samples noisy graphs from a uniform distribution
    X = torch.randint(high=input_dim_X, size=(n_samples, max_nodes)).to(device)
    X = torch.nn.functional.one_hot(X, num_classes=input_dim_X)
    E = torch.randint(high=input_dim_E, size=(n_samples, max_nodes, max_nodes)).to(device)
    E = torch.nn.functional.one_hot(E, num_classes=input_dim_E)
    A = torch.randint(high=input_dim_A, size=(n_samples, max_nodes, input_size_A)).to(device)
    A = torch.nn.functional.one_hot(A, num_classes=input_dim_A)
    L = torch.randint(high=input_dim_L, size=(n_samples, max_nodes, input_size_L)).to(device)
    L = torch.nn.functional.one_hot(L, num_classes=input_dim_L)
    T = torch.randint(high=input_dim_T, size=(n_samples, max_nodes, input_size_T)).to(device)
    T = torch.nn.functional.one_hot(T, num_classes=input_dim_T)
    y = torch.zeros((n_samples, input_dim_y)).to(device)

    # Angle masks
    E = mirror(E)
    A_mask = get_triplet_mask(E.argmax(-1).long(), input_size_A).unsqueeze(-1)
    L_mask = get_line_mask(E.argmax(-1).long(), input_size_L).unsqueeze(-1)
    T_mask = get_twist_mask(E.argmax(-1).long(), input_size_T).unsqueeze(-1)

    # Mirror edge features to ensure graph is undirected and mask unused nodes and angles
    X = X * X_mask
    E = E * E_mask
    A = A * A_mask
    L = L * L_mask
    T = T * T_mask

    return X, E, A, L, T, y, node_mask

def sample_reverse(con_diffusion, cat_diffusion, model, t, x_t):
    # Unpack the data sample and compute masks
    X, E, A, L, T, y, node_mask = x_t
    X, E = X.argmax(dim=-1).long(), E.argmax(dim=-1).long()
    A, L, T = A.argmax(dim=-1).long(), L.argmax(dim=-1).long(), T.argmax(dim=-1).long()
    X_mask = node_mask # B, N
    E_mask = node_mask.unsqueeze(-1).transpose(1,2) * node_mask.unsqueeze(-1) # B, N, N
    device = X.device

    # Prepare timestep t for the batch
    t = (torch.ones(X.shape[0]) * t).long().to(device)
    y_t = torch.hstack((y[:, :-1], t.unsqueeze(-1)))

    # Reverse one step in the diffusion process for X, E
    X_0, E_0, A_0, L_0, T_0, _ = model(X, E, A, L, T, y_t, node_mask)
    X = cat_diffusion.reverse_process(X, X_0, t, X_mask, model.module.input_dims['X'])
    E = cat_diffusion.reverse_process(E, E_0, t, E_mask, model.module.input_dims['E'])
    E = mirror(E)
    
    # Compute angle masks and reverse one step in the diffusion process for A, L, T
    A_mask = get_triplet_mask(E.argmax(-1).long(), model.module.input_sizes['A'])
    L_mask = get_line_mask(E.argmax(-1).long(), model.module.input_sizes['L'])
    T_mask = get_twist_mask(E.argmax(-1).long(), model.module.input_sizes['T'])
    A = cat_diffusion.reverse_process(A, A_0, t, A_mask, model.module.input_dims['A'])
    L = cat_diffusion.reverse_process(L, L_0, t, L_mask, model.module.input_dims['L'])
    T = cat_diffusion.reverse_process(T, T_0, t, T_mask, model.module.input_dims['T'])

    return X, E, A, L, T, y, node_mask

def sample_mols(x_0, dataset):
    # Unpack the data sample
    X, E, A, L, T, _, node_mask = x_0
    # Get atom types, bond types and angles
    X, E = X.argmax(-1), E.argmax(-1)
    A = (A.argmax(-1) * 5 - 2.5).clamp(min=-0, max=180).deg2rad()
    L = (L.argmax(-1) * 5 - 2.5).clamp(min=0, max=360).deg2rad() - torch.pi
    T = (T.argmax(-1) * 5 - 2.5).clamp(min=0, max=360).deg2rad() - torch.pi
    # Get largest fragments of the molecules
    graphs = prepare_angle_graph(X, E, A, L, T, node_mask)
    mols = [mol_from_graph(x, e, dataset.decode_atom, dataset.decode_bond) for x, e, _, _, _ in graphs]
    mols = [(Chem.rdmolops.GetMolFrags(mol, asMols=False, sanitizeFrags=False), idx) for mol, idx in mols]

    # Get indices for the atoms in the largest fragments
    indices = [(max(atoms, key=lambda x: len(x)), idx) for atoms, idx in mols]
    indices = [[key for key, val in idx.items() if val in atoms] for atoms, idx in indices]

    # Remove unused atoms in the graphs
    graphs_new = []
    for (x, e, a, l, t), idx in zip(graphs, indices):
        # Upper triangular part of E
        triu = torch.triu_indices(e.shape[0], e.shape[1], 1)
        # Indexes for keeping
        keep_idx = torch.zeros((e.shape[0], ), device=e.device)
        keep_idx[idx] = 1
        keep_idx = keep_idx.bool()
        # Getting the edges for the largest fragment as a subgraph
        e[triu[1], triu[0]] = e[triu[0], triu[1]]
        edge_index = torch.stack(torch.where(e >= 0))
        edge_attr = e.ravel()
        edge_index, edge_attr = pyg.utils.subgraph(keep_idx, edge_index=edge_index, edge_attr=edge_attr, relabel_nodes=True, num_nodes=len(keep_idx))
        e = pyg.utils.to_dense_adj(edge_index=edge_index, edge_attr=edge_attr).squeeze(0)
        e = e * ~torch.eye(e.shape[0], device=e.device).bool()
        # Reduce graph to only the largest fragment
        graphs_new.append([x[keep_idx], e, a[keep_idx], l[keep_idx], t[keep_idx]])

    # Convert largest fragments into molecules
    mols = [mol_from_graph(x, e, dataset.decode_atom, dataset.decode_bond) for x, e, _, _, _ in graphs_new]
    # Compute 3D positions and add to the molecule
    for (mol, nodes_idx), (X, E, A, L, T) in zip(mols, graphs_new):
        conf = Conformer(mol.GetNumAtoms())
        conf = mol.AddConformer(conf)
        conf = mol.GetConformer(conf)
        zmatrix = get_zmatrix(X.cpu() - 1, E.cpu(), A.cpu(), L.cpu(), T.cpu(), dataset.bond_lengths)
        pos = nerf(zmatrix[:, 1:])
        for i in range(mol.GetNumAtoms()):
            x, y, z = pos[i]
            conf.SetAtomPosition(nodes_idx[int(zmatrix[i,0])], Point3D(x.item(), y.item(), z.item()))
            
    return [mol for mol, _ in mols]
