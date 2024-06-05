import torch
import torch_geometric as pyg
import numpy as np

def to_dense(X, edge_index, edge_attr, batch):
    """
        Convert graphs to dense batch representation and encode no-edge feature.

        Parameters
        ---
        X           : Nodes.
        edge_index  : List of edges.
        edge_attre  : Edge features.
        batch       : List indicating which nodes are related to which graphs.

        Returns
        ---
        X           : The nodes in the graph.
        E           : The dense adjacency matrix of the graph.
        node_mask   : The mask used for masking node in the graph.
    """
    # Convert node features to batch representation
    X, node_mask = pyg.utils.to_dense_batch(x=X, batch=batch)
    # Remove self loops
    edge_index, edge_attr = pyg.utils.remove_self_loops(edge_index, edge_attr)
    # Create dense adjacency matrix with edge features
    max_nodes = X.size(1)
    E = pyg.utils.to_dense_adj(edge_index=edge_index, edge_attr=edge_attr, batch=batch, max_num_nodes=max_nodes)

    # Encode no edge attribute
    E = encode_no_edge(E)

    # Mask unused nodes and return graph and node mask
    X_mask = node_mask.float()
    for _ in range(len(X.shape) - len(node_mask.shape)):
        X_mask = X_mask.unsqueeze(-1)
    E_mask = X_mask.unsqueeze(-1)
    E_mask = E_mask * E_mask.transpose(1, 2)

    X = X * X_mask
    E = E * E_mask

    return X, E, node_mask

def encode_no_edge(E):
    """
        Encodes no-edge features in adjacency matrix.

        Parameters
        ---
        E   : Dense adjacency matrix.

        Returns
        ---
        E   : Dense adjacency matrix with no-edge encoded.
    """
    # Check for one-hot encoded adjacency matrix
    one_hot = len(E.shape) > 3
    # Get mask indicating no edge
    no_edge = E.sum(dim=-1) == 0 if one_hot else E == 0
    # Set no edge attribute
    if one_hot:  E[...,0][no_edge] = 1
    else: E[no_edge] = 1

    return E

def mirror(E):
    """
        Use only upper triangular part for undirected graphs and mirror to lower triangular part.

        Parameters
        ---
        E   : Dense adjacency matrix.

        Returns
        ---
        E   : Mirrored dense adjacency matrix.
    """
    # Get inidices for upper triangular part
    upper_indices = torch.triu_indices(row=E.size(1), col=E.size(2), offset=1)

    # Copy to lower triangular part
    E[:, upper_indices[1], upper_indices[0]] = E[:, upper_indices[0], upper_indices[1]]

    return E

def get_angle_tuples(E):
    """
        Get triplets and 4-tuples for angles in the graph.

        Parameters
        ---
        E           : Dense adjacency matrix.

        Returns
        ---
        triplets    : The triplets for triplet angles.
        tuples_line : The tuples for dihedral angles formed by lines.
        tuples_twist: The tuples for dihedral angles formed by Y-shapes.
    """
    # Convert dense adjacency matrix to list of edges
    edges = torch.stack(torch.where(E > 1)).transpose(1, 0).cpu().numpy()

    # Compute triplets
    triplets = []
    for e1 in edges:
        for e2 in edges:
            if e1[0] == e2[0] and e1[1] != e2[1]: triplets.append([e1[1], e1[0], e2[1]])
    # Remove duplicate triplets
    triplets = np.array([e for e in triplets if e[0] < e[2]], dtype=np.int32)

    # Compute 4-tuples
    tuples_line = []
    tuples_twist = []
    for e1 in edges:
        for e2 in edges:
            if not (e1[1] == e2[0] and e1[0] != e2[1]):
                continue
            for e3 in edges:
                # (a, b), (c, d), (e, f)
                # b == c, a != d, d == e, a != f,
                if e2[1] == e3[0] and e1[0] != e3[1] and e1[1] != e3[1]:
                    tuples_line.append((e1[0], e1[1], e2[1], e3[1]))
                # b == c, a != d, e == c, d != f, a != f
                if e2[0] == e3[0] and e2[1] != e3[1] and e1[0] != e3[1]:
                    tuples_twist.append((e1[0], e1[1], e2[1], e3[1]))

    tuples_twist = np.array(tuples_twist)
    tuples_line = np.array(tuples_line)

    return triplets, tuples_line, tuples_twist

def get_zmatrix(X, E, A, L, T, bond_lengths):
    """
        Our proposed algorithm to convert our spatial graphs into a Z-matrix.

        Parameters
        ---
        X           : Nodes in the graph.
        E           : Edges in the graph.
        A           : Triplet angles in the graph.
        L           : Dihedral angles formed by lines.
        T           : Dihedral angles formed by Y-shapes.
        bond_lengths: Lookup table for bond lengths based on the atom and bond types.

        Returns
        ---
        zmatrix     : The populated Z-matrix for the input graph
    """
    # Mirror the bond lengths lookup table as molecules are undirected graphs.
    indices = np.triu_indices(bond_lengths.shape[0], k=1)
    bond_lengths[indices[1], indices[0]] = bond_lengths[indices[0], indices[1]]
    # Get max index values for the angles.
    A_max, L_max, T_max = A.shape[-1] - 1, L.shape[-1] - 1, T.shape[-1] - 1

    # Get angle tuples - determining the order of the angles in the matrices.
    triplets, tuples_line, tuples_twist = get_angle_tuples(E)
    # Initialise the Z-matrix (Type, atom1, bond length, atom2, triplet angle, atom3, dihedral angle)
    zmatrix = np.zeros((X.shape[0], 7))
    # Insert one atom from the graph into the Z-matrix at a time
    for i in range(X.shape[0]):
        # Case 1 of the Z-matrix
        if i == 0:
            # Get atom connected to the least amount of atoms - finding an endpoint for a triplet
            atom1 = np.argmin((E > 1).sum(-1))
            # Populate entry in the Z-matrix
            zmatrix[i] = [atom1, 0, 0, 0, 0, 0, 0]
            # Remove edges going back to the inserted atom - making sure it is not inserted twice
            E[:, atom1] = 0
            continue

        # Case 2 of the Z-matrix
        if i == 1:
            # Get first inserted atom
            atom1 = int(zmatrix[0,0])
            # Get atom connected to atom1
            atom2 = np.where(E[atom1] > 1)[0][0]
            # Get the bond type between atom1 and atom2
            bond = E[atom1, atom2] - 2
            # Lookup bond length for the given type between the two types of atoms
            bond_length = bond_lengths[X[atom1], X[atom2], bond]
            # Populate entry in the Z-matrix
            zmatrix[i] = [atom2, 0, bond_length, 0, 0, 0, 0]
            # Remove edges going back to the inserted atom - making sure it is not inserted twice
            E[:, atom2] = 0
            continue

        # Case 3 of the Z-matrix
        if i == 2:
            # Get first and second inserted atoms
            atom1, atom2 = int(zmatrix[0,0]), int(zmatrix[1,0])
            # Get atom connected to atom2
            atom3 = np.where(E[atom2] > 1)[0][0]
            # Get the bond type between atom2 and atom3
            bond = E[atom2, atom3] - 2
            # Lookup bond length for the given type between the two types of atoms
            bond_length = bond_lengths[X[atom2], X[atom3], bond]  
            # Get the triplet angle formed by atom1, atom2, and atom3 using atom2 as reference
            angle_index = [i for i, x in enumerate(triplets[triplets[:,1] == atom2]) if (x == [atom1, atom2, atom3]).all() or (x == [atom3, atom2, atom1]).all()]
            angle_index = min(A_max, angle_index[0]) if len(angle_index) > 0 else 0
            angle = A[atom2, angle_index]
            # Populate entry in the Z-matrix
            zmatrix[i] = [atom3, 1, bond_length, 0, angle, 0, 0]
            # Remove edges going back to the inserted atom - making sure it is not inserted twice
            E[:, atom3] = 0
            continue
        
        # Case 4 of the Z-matrix
        # Find unvisited atom connected to a visited atom looping backwards through the Z-matrix
        for j in range(i-1,-1,-1):
            atom3 = int(zmatrix[j,0])
            atom4 = np.where(E[atom3] > 1)[0]
            if len(atom4) > 0:
                atom4 = atom4[0]
                break
        # Special case when the unvisited atom is connected to the first inserted atom
        if j == 0:
            atom3_z = 0
            atom2_z = 1
            atom1_z = 2
        else:
            # Get row numbers for the reference atoms
            atom3_z = j
            atom2_z = int(zmatrix[j,1])
            atom1_z = int(zmatrix[j,3])
        # Get the atom indices for our graph
        atom1 = int(zmatrix[atom1_z, 0])
        atom2 = int(zmatrix[atom2_z, 0])
        
        # Get the bond type between atom3 and atom4
        bond = E[atom3, atom4] - 2
        # Lookup bond length for the given type between the two types of atoms
        bond_length = bond_lengths[X[atom3], X[atom4], bond]

        # Get the triplet angle formed by atom2, atom3, and atom4 using atom3 as reference
        angle_index = [i for i, x in enumerate(triplets[triplets[:,1] == atom3]) if (x == [atom2, atom3, atom4]).all() or (x == [atom4, atom3, atom2]).all()]
        angle_index = min(A_max, angle_index[0]) if len(angle_index) > 0 else 0
        angle = A[atom3, angle_index]

        # Special case for atom4 being connected to the second inserted atom and when the dihedral angle forms a Y-shape.
        is_line = True
        if atom1_z == atom2_z:
            is_line = False
            atom1_z = j+1
            atom1 = int(zmatrix[j+1, 0])
        tuples = tuples_line if is_line else tuples_twist
        d_max = L_max if is_line else T_max


        # Get the dihedral angle formed by atom1, atom2, atom3, and atom4 using atom1 as reference
        comp = [atom1, atom2, atom3, atom4] if is_line else [atom1, atom3, atom2, atom4]
        dangle_index = [i for i, x in enumerate(tuples[tuples[:,0] == atom1]) if (x == comp).all()]
        if len(dangle_index) > 0:
            dangle_index = min(d_max, dangle_index[0])
            dangle = L[atom1, dangle_index] if is_line else -T[atom1, dangle_index]
        else: dangle = 0.

        # Populate entry in the Z-matrix
        zmatrix[i] = [atom4, atom3_z, bond_length, atom2_z, angle, atom1_z, dangle]
        # Remove edges going back to the inserted atom - making sure it is not inserted twice
        E[:, atom4] = 0

    return zmatrix

def nerf(zmat):
    """
        The NeRF algorithm convert the Z-matrix into Cartesian coordinates.

        Parameters
        ---
        zmat    : The Z-matrix for a molecule.

        Returns
        ---
        cart    : The Cartesian coordinates for the atoms.
    """
    # Initialise list of Cartesian coordinates
    cart = []
    # Compute x, y, z for one atom at a time
    for i, row in enumerate(zmat):
        # Unpack the values for the row in the zmatrix
        a1, bl, a2, angle, a3, dangle = row
        a1, a2, a3 = int(a1), int(a2), int(a3)

        # First case of NeRF
        if i == 0: 
            # Put the first atom in origin (0, 0, 0)
            cart.append(np.array([0.0, 0.0, 0.0]))
            continue

        # Second case of NeRF
        if i == 1: 
            # Put the second atom by translating the bond length from origin on the x-axis
            cart.append(np.array([bl, 0.0, 0.0]))
            continue

        # Third case of NeRF
        if i == 2:
            # Put the third atom by moving the second atom in xy-plane using the triplet angle and bond length
            cart = np.array(cart)
            x = cart[a1, 0] + bl * np.cos(np.pi - angle)
            y = bl * np.sin(angle)
            cart = np.append(cart, np.array([x, y, 0.0])[None, :], axis=0)
            continue
        
        # Forth case of NeRF
        # Get spherical representation of the next atom
        cx = bl * np.cos(np.pi - angle)
        cy = bl * np.cos(dangle) * np.sin(angle)
        cz = bl * np.sin(np.pi + dangle) * np.sin(angle)

        # Get coordinates of reference atoms
        a, b, c = cart[a3], cart[a2], cart[a1]

        # Compute transition matrix based on reference atoms
        ab = b - a
        ab = ab / np.linalg.norm(ab) if np.linalg.norm(ab) != 0. else ab
        bc = c - b
        bc = bc / np.linalg.norm(bc) if np.linalg.norm(bc) != 0. else bc

        n = np.cross(ab, bc)
        n = n / np.linalg.norm(n) if np.linalg.norm(n) != 0. else n
        m = np.cross(n, bc)
        m = m / np.linalg.norm(m) if np.linalg.norm(m) != 0. else m

        M = np.stack([bc, m, n]).swapaxes(0, 1)

        # Apply transition matrix to the spherical representation to get relative positition to reference atom
        # Add relative position to coordinates of the reference atom
        pos = (M @ np.array([cx, cy, cz])) + c
        cart = np.append(cart, pos[None, :], axis=0)

    return np.array(cart)

def get_mask_from_counts(max_angles, angles_count, device, bs, n):
    """
        Converts counts of angles into an angle mask.

        Parameters
        ---
        max_angles      : Max number of angles for each reference atom.
        angles_count    : The counts of the angles for each reference atom.
        device          : The CUDA device to create the mask on.
        bs              : The batch size.
        n               : Max number of nodes in the batched graphs.

        Returns
        ---
        angle_mask      : The angle mask used for masking angles.
    """
    # Prepare an arange for the mask
    arange = torch.arange(max_angles, device=device).unsqueeze(0).expand(n, -1).unsqueeze(0).expand(bs, n, -1)
    # Create the angle masked using the arange and the counts
    angle_mask = arange < angles_count.unsqueeze(2)
    return angle_mask

def get_triplet_mask(E, max_angles):
    """
        Counts the number of possible triplet angles and converts to a mask.

        Parameters
        ---
        E           : The dense adjacency matrix.
        max_angles  : Max number of angles for each reference atom.

        Returns
        ---
        mask        : The angle mask used for masking angles.
    """
    # Get connections in the graph
    Ex = (E > 1).long()
    # Get counts as possible combinations from a reference atom
    counts = Ex.sum(-1) * (Ex.sum(-1) - 1) / 2
    # Convert counts to mask
    mask = get_mask_from_counts(max_angles, counts, E.device, E.size(0), E.size(1))
    return mask.long()

def get_line_mask(E, max_angles):
    """
        Counts the number of possible dihedral angles formed by a line and converts to a mask.

        Parameters
        ---
        E           : The dense adjacency matrix.
        max_angles  : Max number of angles for each reference atom.

        Returns
        ---
        mask        : The angle mask used for masking angles.
    """
    # Get connections in the graph
    Ex = (E > 1).long()
    # Get diagonal mask for E
    diag = (~torch.eye(Ex.shape[1], device=Ex.device).bool()).long()
    # Get neighbour entries for each reference atom
    neigh = (Ex.sum(-1) - 1).clamp(min=0).unsqueeze(-1) * Ex
    # Get counts of angles as E^3 without diagonal and returning to neighbours
    counts = ((Ex.float() @ Ex.float() * diag) @ Ex.float() * diag - neigh).long().sum(-1)
    # Convert counts to mask
    mask = get_mask_from_counts(max_angles, counts, E.device, E.size(0), E.size(1))
    return mask.long()

def get_twist_mask(E, max_angles):
    """
        Counts the number of possible dihedral angles formed by a Y-shape and converts to a mask.

        Parameters
        ---
        E           : The dense adjacency matrix.
        max_angles  : Max number of angles for each reference atom.

        Returns
        ---
        mask        : The angle mask used for masking angles.
    """
    # Get connections in the graph
    Ex = (E > 1).long()
    # Get number of twist for a central atom
    num_twists = (Ex.sum(-1) - 1) * (Ex.sum(-1) - 2)
    # Get counts as sum of all twist for all central atom reachable in 1 step
    counts = (Ex * num_twists.unsqueeze(-1)).sum(-1)
    # convert counts to mask
    mask = get_mask_from_counts(max_angles, counts, E.device, E.size(0), E.size(1))
    return mask.long()
