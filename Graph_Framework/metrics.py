from rdkit import Chem
from tqdm import tqdm
from rdkit import rdBase
rdBase.DisableLog('rdApp.*')

from utils.mol_utils import *
from utils.graph_utils import *


def load_smiles(smiles_path):
    """
        Load dataset smiles from path.

        Parameters
        ---
        smiles_path : Path to file with SMILES.

        Returns
        ---
        smiles      : A list of the loaded SMILES.
    """
    with open(smiles_path, 'r') as f:
        smiles = f.read().splitlines()

    return smiles

def compute_validity(mols):
    """
        Computes validity using RDKit.

        Parameters
        ---
        mols        : RDKit molecules to check validity of.

        Returns
        ---
        valid       : List of SMILES for valid molecules.
        all_smiles  : List of SMILES for all mulecules.
    """
    # Initialise lists
    valid = []
    all_smiles = []

    # Iterate every molecule
    for mol in mols:
        try:
            # Get largest fragment if the sample contains multiple molecules
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            # Sanitise the molecule
            Chem.SanitizeMol(largest_mol)
            # Generate unique SMILES string for the molecule and add it to the lists
            smiles = Chem.MolToSmiles(largest_mol)
            valid.append(smiles)
            all_smiles.append(smiles)
        except (ValueError, Chem.rdchem.AtomValenceException, Chem.rdchem.KekulizeException):
            all_smiles.append(None)

    return valid, all_smiles

def compute_uniqueness(smiles):
    """
        Computes unique set of SMILES.

        Parameters
        ---
        smiles  : List of SMILES.

        Returns
        ---
        smiles  : Unique set of SMILES.
    """
    return list(set(smiles))

def compute_novelty(smiles, dataset_smiles, remove_h=False):
    """
        Computes the list of novel samples based on SMILES from the traiing dataset.

        Parameters
        ---
        smiles          : List of SMILES to check for novelty.
        dataset_smiles  : List of dataset SMILES to match against.
        remove_h        : Whether to compute novelty with or without hydrogen atoms.

        Returns
        ---
        smiles          : List of novel SMILES.
    """
    # Remove hydrogen atoms in the SMILES
    if remove_h: 
        dataset_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in dataset_smiles]
        smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in smiles if Chem.MolFromSmiles(x) is not None]
    
    # Convert dataset SMILES to dictionary for faster matching
    dset_smiles = dict((s, 1) for s in dataset_smiles)

    # Compute novelty
    return [x for x in smiles if x not in dset_smiles]

def compute_dataset_smiles(dataloader, decode_atom, decode_bond, save_path):
    """
        Make a pass over the dataset and compute its SMILES strings.

        Parameters
        ---
        dataloader  : The dataloader for the graphs to convert to SMILES.
        decode_atom : Dictionary to decode node features to atom types.
        decode_bond : Dictionary to decode edge features to bond types.
        save_path   : Path to save the SMILES to.
    """
    print("Computing Dataset SMILES")
    dataset_smiles = []
    # Iterate every batch of graphs in the dataloder
    for data in tqdm(dataloader):
        # Convert graph to node features and edge features
        x, e, node_mask = to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        # Convert graph to molecule representation
        if len(x.shape) > 2: x, e = x.argmax(-1), e.argmax(-1)
        mols = [mol_from_graph(x_i, e_i, decode_atom, decode_bond)[0] for x_i, e_i in prepare_graph(x, e, node_mask)]
        # Compute valid set of SMILES for the dataloder
        valid, _ = compute_validity(mols)
        dataset_smiles.extend(valid)
    
    # Get unique list of SMILES and save them
    dataset_smiles = list(set(dataset_smiles))
    with open(save_path, 'w') as f:
        for smiles in [s for s in dataset_smiles if s != '']:
            f.write(smiles + '\n')
