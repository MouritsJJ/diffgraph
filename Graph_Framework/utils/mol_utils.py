from rdkit import Chem
from rdkit.Chem import Draw

def save_molecules(mols, path):
    """
        Save molecules using RDKit.

        Parameters
        ---
        mols    : Molecules to save as images.
        path    : Where to save the images.
    """
    # Put molecules into a grid image
    img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(400, 400))
    # Save the image
    img.save(path)

def prepare_graph(X, E, node_mask):
    """
        Prepares a batch of graphs for conversion to molecules.

        Parameters
        ---
        X           : Batch of nodes in graphs.
        E           : Batch of edges in graphs.
        node_mask   : The mask used for checking the number of nodes in each batched graph.

        Returns
        ---
        graphs      : List of graphs, where unused nodes are removed.
    """
    # Initialise list of graphs
    graphs = []
    # Iterate every graph
    for x, e, mask in zip(X, E, node_mask):
        nodes = mask.long().sum().item()
        # Remove unused nodes
        graphs.append((x[:nodes], e[:nodes, :nodes]))
        
    return graphs

def prepare_angle_graph(X, E, A, L, T, node_mask):
    """
        Prepares a batch of graphs with spatial information for conversion to molecules.

        Parameters
        ---
        X           : Batch of nodes in graphs.
        E           : Batch of edges in graphs.
        A           : Batch of tiplet angles.
        L           : Batch of dihedral angles formed by lines.
        T           : Batch of dihedral angles formed by Y-shapes.
        node_mask   : The mask used for checking the number of nodes in each batched graph.

        Returns
        ---
        graphs      : List of graphs, where unused nodes are removed.
    """
    graphs = []
    # Iterate every graph
    for x, e, a, l, t, mask in zip(X, E, A, L, T, node_mask):
        nodes = mask.long().sum().item()
        # Remove unused nodes
        graphs.append((x[:nodes], e[:nodes, :nodes], a[:nodes], l[:nodes], t[:nodes]))
        
    return graphs

def mol_from_graph(x, e, decode_atom, decode_bond):
    """
        Converts a graph into a RDKit molecule.

        Parameters
        ---
        x           : Nodes in the graph.
        e           : Edges in the graph.
        decode_atom : Dictionary to decode node feature as atom type.
        decode_bond : Dictionary to decode edge feature as bond type.

        Returns
        ---
        mol         : The RDKit molecule.
        nodes_idx   : Dictionary converting from indices in the input graph to atom indices in the output molecule.
    """
    # Create raw molecule
    mol = Chem.RWMol()
    
    # Add atoms to molecule
    nodes_idx = {}
    for ia, a in enumerate(x):
        atom = Chem.Atom(decode_atom[a.item()])
        idx = mol.AddAtom(atom)
        nodes_idx[ia] = idx
    
    # Add bonds between atom in molecule
    for ix, row in enumerate(e):
        for iy, bond in enumerate(row):
            bond = decode_bond[bond.item()]
            if iy <= ix or bond is None:
                continue
            mol.AddBond(nodes_idx[ix], nodes_idx[iy], bond)

    # Return molecule if valid otherwise return empty molecule
    try:
        mol = mol.GetMol()
    except Chem.KekulizeException:
        print("Can't kekulize molecule")
        mol = None
        
    return mol, nodes_idx
