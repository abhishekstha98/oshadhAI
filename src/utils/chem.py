import logging
from rdkit import Chem
from typing import Optional, Tuple

def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Returns canonical SMILES if valid, else None.
    """
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False) # Isomeric=False for broader matching in MPV, can be True if strict
    except Exception:
        return None

def get_inchikey(smiles: str) -> Optional[str]:
    """
    Returns InChIKey from SMILES.
    """
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None

def standardize_compound(smiles: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (Canonical SMILES, InChIKey).
    """
    canon = canonicalize_smiles(smiles)
    if not canon:
        return None, None
    inchi = get_inchikey(canon)
    return canon, inchi
