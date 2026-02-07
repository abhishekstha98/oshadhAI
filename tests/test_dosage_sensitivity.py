import unittest
import torch
from src.inference.explain import Explainer

class TestDosageSensitivity(unittest.TestCase):
    def setUp(self):
        # Use random weights for demo since we don't have checkpoints in test env
        self.explainer = Explainer()
    
    def test_dosage_monotonicity(self):
        # Testing monotonicity of combined influences
        smiles = ["CCO", "CCN"]
        
        # Scenario 1: Compound A High Dose, Compound B Low Dose
        res1 = self.explainer.explain(smiles, weights=[0.9, 0.1])
        ci1 = res1['dose_analysis']['combined_influence']
        
        # Scenario 2: Compound A Low Dose, Compound B High Dose
        res2 = self.explainer.explain(smiles, weights=[0.1, 0.9])
        ci2 = res2['dose_analysis']['combined_influence']
        
        self.assertGreater(ci1[smiles[0]], ci2[smiles[0]])
        self.assertGreater(ci2[smiles[1]], ci1[smiles[1]])

    def test_ic50_monotonicity(self):
        smiles = ["CCO", "CCN"]
        
        # Scenario 1: Compound A High Potency (Low IC50), Compound B Low Potency (High IC50)
        res1 = self.explainer.explain(smiles, ic50_list=[0.1, 10.0])
        ci1 = res1['dose_analysis']['combined_influence']
        
        # Scenario 2: Compound A Low Potency (High IC50), Compound B High Potency (Low IC50)
        res2 = self.explainer.explain(smiles, ic50_list=[10.0, 0.1])
        ci2 = res2['dose_analysis']['combined_influence']
        
        self.assertGreater(ci1[smiles[0]], ci2[smiles[0]])
        self.assertGreater(ci2[smiles[1]], ci1[smiles[1]])

if __name__ == '__main__':
    unittest.main()
