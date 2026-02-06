import torch
from rdkit import Chem
from typing import Optional, Tuple
from rdkit import RDLogger

# Suppress RDKit info/warnings
RDLogger.DisableLog('rdApp.*')

# Allowed atom types for biological grounding (ChEMBL subset extended)
ATOM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']

def one_of_k_encoding(x, allowable_set) -> list:
    if x not in allowable_set:
        x = allowable_set[-1] # Catch-all
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom) -> torch.Tensor:
    """
    Featurize atom: Symbol, Degree, NumHs, Valence, Aromaticity.
    Total Dim: 43 + 11 + 5 + 6 + 1 = 66
    (Adjusted from previous 63 to be precise with lists)
    """
    return torch.tensor(one_of_k_encoding(atom.GetSymbol(), ATOM_LIST) +
                        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                        one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                        [atom.GetIsAromatic()], dtype=torch.float)

def smile_to_graph(smile: str) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Converts SMILES to (x, edge_index, edge_attr).
    Strict validation.
    """
    if not smile: return None
    mol = Chem.MolFromSmiles(smile)
    if mol is None: return None
    
    # Validation constraint: Minimum atoms
    if mol.GetNumAtoms() < 1: return None

    # Node Features
    x_list = []
    for atom in mol.GetAtoms():
        x_list.append(atom_features(atom))
    x = torch.stack(x_list)

    # Edge Features
    rows, cols, edge_feats = [], [], []
    
    if len(mol.GetBonds()) > 0: 
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Feature: Bond type (One-hot 4 types)
            bond_type = bond.GetBondType()
            bt = [0, 0, 0, 0]
            if bond_type == Chem.rdchem.BondType.SINGLE: bt[0] = 1
            elif bond_type == Chem.rdchem.BondType.DOUBLE: bt[1] = 1
            elif bond_type == Chem.rdchem.BondType.TRIPLE: bt[2] = 1
            elif bond_type == Chem.rdchem.BondType.AROMATIC: bt[3] = 1
            else: bt[0] = 1 # Fallback to single
            
            # Bidirectional
            rows += [i, j]
            cols += [j, i]
            edge_feats += [bt, bt]
            
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float) # Float for embeddings
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float)

    return x, edge_index, edge_attr
