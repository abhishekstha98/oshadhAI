from rdkit import Chem
from rdkit.Chem import QED, Lipinski
import logging

class QsarPredictor:
    """
    Predicts general bioactivity potential using structural properties (QSAR).
    Uses QED (Quantitative Estimation of Drug-likeness) and Lipinski compliance as proxies
    when experimental IC50 data is unavailable.
    """
    def predict_bioactivity(self, smiles: str) -> float:
        """
        Estimate bioactivity potential (0.0 - 1.0).
        Logic: QED * Lipinski_Compliance_Factor
        """
        if not smiles:
            return 0.0
            
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return 0.0
            
        try:
            # 1. Calculate QED (0-1)
            # Weighted sum of desirability functions for MW, logP, etc.
            qed_score = QED.qed(mol)
            
            # 2. Lipinski Rule of 5 Compliance
            # Violations reduce the confidence in "drug-like" bioactivity
            violations = 0
            if Lipinski.NumHDonors(mol) > 5: violations += 1
            if Lipinski.NumHAcceptors(mol) > 10: violations += 1
            if Lipinski.rdMolDescriptors.CalcExactMolWt(mol) > 500: violations += 1
            # LogP is already part of QED, but explicit check is common
            
            # Penalty factor: 1.0 for 0 violations, 0.8 for 1, etc.
            compliance_factor = max(0.5, 1.0 - (violations * 0.15))
            
            # Final Proxy Score
            bioactivity_proxy = qed_score * compliance_factor
            
            return round(bioactivity_proxy, 3)
            
        except Exception as e:
            logging.warning(f"QSAR prediction failed for {smiles}: {e}")
            return 0.0
