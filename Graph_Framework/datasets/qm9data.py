import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import subgraph
from torch_geometric.datasets import QM9
from tqdm import tqdm
from rdkit import Chem
import numpy as np

class QM9Data(InMemoryDataset):
    """
        An extension to the QM9 dataset from PyG used for preprocessing the data for the model
    """
    def __init__(self, config) -> None:
        """
            Parameters
            ---
            config  : Dictionary containing entries: path, remove_h, categorical_types and categorical_angles
        """
        self.dataset = QM9(root=config["path"])
        self.remove_h = config['remove_h']
        self.categorical_types = config['categorical_types']
        self.categorical_angles = config['categorical_angles'] if 'categorical_angles' in config else None

        # Node probabilities used for generation
        if self.remove_h:
            self.node_probs = torch.tensor([0, 2.2930e-05, 3.8217e-05, 6.8791e-05, 2.3695e-04, 9.7072e-04,
                                            0.0046472, 0.023985, 0.13666, 0.83337])
        else:
            self.node_probs = torch.tensor([0, 0, 0, 1.5287e-05, 3.0574e-05, 3.8217e-05,
                                            9.1721e-05, 1.5287e-04, 4.9682e-04, 1.3147e-03, 3.6918e-03, 8.0486e-03,
                                            1.6732e-02, 3.0780e-02, 5.1654e-02, 7.8085e-02, 1.0566e-01, 1.2970e-01,
                                            1.3332e-01, 1.3870e-01, 9.4802e-02, 1.0063e-01, 3.3845e-02, 4.8628e-02,
                                            5.4421e-03, 1.4698e-02, 4.5096e-04, 2.7211e-03, 0.0000e+00, 2.6752e-04])

        # Bond lengths
        # https://chem.libretexts.org/Ancillary_Materials/Reference/Reference_Tables/Atomic_and_Molecular_Properties/A3%3A_Covalent_Radii
        self.bond_lengths = np.array(
            [[[0.62, 0, 0, 0.62], [1.07, 0, 0, 1.07], [1.02, 0, 0, 1.02], [0.97, 0, 0, 0.97], [0.88, 0, 0, 0.88]],
            [[0, 0, 0, 0], [1.52, 1.34, 1.2, 1.52], [1.47, 1.27, 1.14, 1.47], [1.42, 1.24, 1.13, 1.42], [1.33, 1.26, 1.13, 1.33]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [1.42, 1.2, 1.08, 1.42], [1.37, 1.17, 1.07, 1.37], [1.28, 1.19, 1.07, 1.28]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1.32, 1.14, 1.06, 1.32], [1.23, 1.16, 1.06, 1.23]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1.14, 1.18, 1.06, 1.14]]]
        )
        if self.remove_h: self.bond_lengths = self.bond_lengths[1:, 1:]

        # Encoding of atoms in QM9
        # { 1:'H', 6:'C', 7:'N', 8:'O', 9:'F' }
        self.encode_atoms = { 1:1, 6:2, 7:3, 8:4, 9:5 }

        # Decoding the atoms based on settings
        if self.categorical_types:
            if self.remove_h: self.decode_atom = { 1:6, 2:7, 3:8, 4:9 }
            else: self.decode_atom = { 1:1, 2:6, 3:7, 4:8, 5:9 }
        else:
            if self.remove_h: self.decode_atom = { 0:6, 1:7, 2:8, 3:9 }
            else: self.decode_atom = { 0:1, 1:6, 2:7, 3:8, 4:9 }

        # Decoding bond types based on the settings
        if self.categorical_types: self.decode_bond = {
            0: None,
            1: None,
            2: Chem.rdchem.BondType.SINGLE,
            3: Chem.rdchem.BondType.DOUBLE,
            4: Chem.rdchem.BondType.TRIPLE,
            5: Chem.rdchem.BondType.AROMATIC
        }
        else: self.decode_bond = {
            0: None,
            1: Chem.rdchem.BondType.SINGLE,
            2: Chem.rdchem.BondType.DOUBLE,
            3: Chem.rdchem.BondType.TRIPLE,
            4: Chem.rdchem.BondType.AROMATIC
        }

        super().__init__(config["path"])
        self.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        # File names for raw QM9 dataset
        return self.dataset.raw_file_names

    @property
    def processed_file_names(self):
        # File names for custom processed dataset
        path = ''
        path += 'cat' if self.categorical_types else 'con'
        if self.categorical_angles is not None:
            path += '_cat' if self.categorical_angles else '_con'
        path += '_noH' if self.remove_h else '_H'
        
        return [path + '.pt']

    def download(self):
        self.dataset.download()

    def process(self):
        """
            Processes the dataset based on the input settings.
        """
        # Converts graphs based on settings
        print("Custom processing...")
        data_list = [self.convert(g) for g in tqdm(self.dataset)]
        # Saves processed graphs
        self.save(data_list, self.processed_paths[0])
        
    def convert(self, g):
        """
            Converts a graph into our graph representation with and without spatial information.

            Parameters
            ---
            g   : Graph to convert

            Returns
            ---
            data: A PyG data object of the converted graph.
        """
        # Get one-hot encodings of atom types
        x = torch.tensor([self.encode_atoms[mol.item()] for mol in g.z]).float()
        # Remove hydrogen atoms
        if self.remove_h:
            # Atom types
            keep_idx = x > 1
            x = x[keep_idx] - 1
            if not self.categorical_types: x = F.one_hot(x.long() - 1, num_classes=len(self.decode_atom)).float()

            # Bond types
            edge_index, edge_attr = subgraph(keep_idx, g.edge_index, g.edge_attr, relabel_nodes=True, num_nodes=len(keep_idx))

            # Coordinates
            pos = g.pos[keep_idx]

        # Process graph keeping hydrogen atoms
        else:
            # Bond types
            edge_index, edge_attr = g.edge_index, g.edge_attr

            # Coordinates
            pos = g.pos
        
        # One-hot encode atom types for continuous models
        if not self.categorical_types: x = F.one_hot(x.long() - 1, num_classes=len(self.decode_atom)).float()

        # Process bond types for categorical models
        if self.categorical_types:
            # Get category from one-hot encoded bonds
            edge_attr = torch.argmax(edge_attr, dim=1) + 2
        # Process bond types for continuous models
        else:
            # Make space for no-bond attribute in the encoding of bonds
            edge_attr0 = torch.zeros((edge_attr.size(0), 1))
            edge_attr = torch.cat((edge_attr0, edge_attr), dim=1)

        # Do not use labels for this dataset
        y = torch.zeros((1, 0), dtype=torch.float)

        # Process angles for graphs with spatial information
        if self.categorical_angles is not None:
            # Get angles in the graph
            angles = torch.tensor(self.compute_triplet_angles(pos, edge_index)).float()
            line, twist = self.compute_dihedral_angles(pos, edge_index)
            line, twist = torch.tensor(line).float(), torch.tensor(twist).float()
            
            # Create PyG data object representing a graph
            data = pyg.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, angles=angles, langles=line, tangles=twist)

        # Create converted PyG data object without spatial information
        else:
            # Create PyG data object representing a graph
            data = pyg.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            

        return data

    def compute_triplet_angles(self, pos, edge_index):
        """
            Computes the internal angles for all connected triplets in the graph.

            Parameters
            ---
            pos         : The Cartesian coordinates of the atoms in the molecule.
            edge_index  : The list of edges in the graph.

            Returns
            ---
            angle_mat   : A matrix with the triplet angles in rows based on reference atoms.
        """
        # Get 3D positions and convert edges to pairs
        pos = np.array(pos)
        edges = np.array(edge_index).transpose(1,0)

        # Find triplets in the molecule
        triplets = []
        for e1 in edges:
            for e2 in edges:
                if e1[0] == e2[0] and e1[1] != e2[1]: triplets.append([e1[1], e1[0], e2[1]])
        # Remove duplicate triplets
        triplets = np.array([e for e in triplets if e[0] < e[2]], dtype=np.int32)
        if triplets.size == 0:
            return np.zeros((pos.shape[0], 10))

        # Compute angles for each triplet
        vectors = np.array([[pos[t[0]] - pos[t[1]], pos[t[2]] - pos[t[1]]] for t in triplets], dtype=np.float64)
        angles = np.array([np.arccos(np.dot(v[0], v[1]) / ( np.linalg.norm(v[0]) * np.linalg.norm(v[1]) )) for v in vectors])
        angles = np.clip(np.degrees(angles), a_min=0, a_max=179) // 5 + 1 if self.categorical_angles else (angles - 0.5 * np.pi) / (0.5 * np.pi)

        # Get number of angles for each reference atom
        lens = np.array(np.unique(triplets[:,1], return_counts=True)).transpose(1,0)

        # Convert to angle matrix (N, 10)
        angle_mat = np.zeros((pos.shape[0], 10))
        count = 0
        for s, l in lens:
            for i in range(l):
                # angle_mat[angle[1], :, i] = 1
                angle_mat[s, i] = angles[count+i] if not np.isnan(angles[count+i]) else 0.0
            
            count += l
        
        return angle_mat

    def compute_dihedral_angle(self, pos, atoms, line=True):
        """
            Compute a single dihedral angle in the graph.

            Parameters
            ---
            pos     : The Cartesian coordinates of the atoms for the dihedral angle.
            atoms   : The 4 atoms forming a dihedral angle.
            line    : Whether the 4 atoms form a line or Y-shape dihedral angle.

            Returns
            ---
            angle   : The dihedral angle formed by the 4 atoms.
        """
        # Positions
        ap, bp, cp, dp = pos[atoms[0]], pos[atoms[1]], pos[atoms[2]], pos[atoms[3]]
        # Vectors
        v1 = bp - ap
        v2 = cp - bp
        v3 = dp - cp if line else dp - bp
        # Normal vectors
        n1 = np.cross(v1, v2)
        n1 = n1 / np.linalg.norm(n1)
        n2 = np.cross(v2, v3)
        n2 = n2 / np.linalg.norm(n2)
        # Angles
        m1 = np.cross(n1, v2 / np.linalg.norm(v2))
        sim1 = np.dot(n1, n2)
        sim2 = np.dot(m1, n2)
        angle = np.arctan2(sim2, sim1)

        if self.categorical_angles:
            angle = np.clip(np.degrees(angle + np.pi), a_min=0, a_max=360) // 5 + 1 if not np.isnan(angle) else 0
        else:
            angle = angle / np.pi if not np.isnan(angle) else 0.0

        return angle

    def compute_dihedral_angles(self, pos, edge_index):
        """
            Computes all the dihedral angles in the graph.

            Parameters
            ---
            pos             : The Cartesian coordinates of the atoms in the molecule.
            edge_index      : The list of edges in the graph.

            Returns
            ---
            angle_mat_line  : The dihedral angles, where the angle is based on 4 atoms forming a line.
            angle_mat_twist : The dihedral angles, where the angle is based on 4 atoms forming a Y-shape.
        """
        # Get 3D positions and convert edges to pairs
        pos = np.array(pos)
        edges = np.array(edge_index).transpose(1,0)

        # Find 4-tuplets in the molecule
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
        if tuples_line.size == 0 and tuples_twist.size == 0:
            return np.zeros((pos.shape[0], 28)), np.zeros((pos.shape[0], 48))

        # Compute angles for each tuple
        angles_line, angles_twist = [], []
        for t in tuples_line:
            angles_line.append(self.compute_dihedral_angle(pos, t, True))
        for t in tuples_twist:
            angles_twist.append(self.compute_dihedral_angle(pos, t, False))

        angles_line, angles_twist = np.array(angles_line), np.array(angles_twist)
        counts_line = np.array(np.unique(tuples_line[:, 0], return_counts=True)).transpose(1, 0) if tuples_line.size > 0 else []
        counts_twist = np.array(np.unique(tuples_twist[:, 0], return_counts=True)).transpose(1, 0) if tuples_twist.size > 0 else []

        # Convert to dihedral angle matrices (N, 28) & (N, 48)
        angle_mat_line, angle_mat_twist = np.zeros((pos.shape[0], 28)), np.zeros((pos.shape[0], 48))
        index = 0
        for c in counts_line:
            for i in range(c[1]):
                angle_mat_line[c[0], i] = angles_line[index + i]
            index += c[1]
        index = 0
        for c in counts_twist:
            for i in range(c[1]):
                angle_mat_twist[c[0], i] = angles_twist[index + i]
            index += c[1]
        
        return angle_mat_line, angle_mat_twist
