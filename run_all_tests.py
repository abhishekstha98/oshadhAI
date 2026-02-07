import sys
import unittest
from pathlib import Path
import logging
import json

# Adjust path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.ingest.db import DB_PATH
from src.inference.explain import Explainer
from src.inference.feedback import FeedbackManager
from src.inference.qsar import QsarPredictor

logging.basicConfig(level=logging.INFO)

class TestAntigravityFunctionalities(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print("\n=== Starting Comprehensive Functional Tests ===")
        print(f"Project Root: {project_root}")
        cls.explainer = Explainer()
        cls.feedback_manager = FeedbackManager()
        cls.qsar = QsarPredictor()

    def test_01_ingestion_artifacts_exist(self):
        """Verify database and model weights exist."""
        print("\n[Test 1] Verifying Ingestion & Training Artifacts...")
        self.assertTrue(Path(DB_PATH).exists(), "Database file missing!")
        self.assertTrue(Path("checkpoints/stage1.pt").exists(), "Stage 1 weights missing!")
        self.assertTrue(Path("checkpoints/stage3.pt").exists(), "Stage 3 weights missing!")
        print("-> Artifacts confirmed.")

    def test_02_standard_inference(self):
        """Test basic scoring functionality."""
        print("\n[Test 2] Testing Standard Inference...")
        # Aspirin + Caffeine
        smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
        result = self.explainer.explain(smiles)
        self.assertIn("plausibility_score", result)
        self.assertIn("decision", result)
        print(f"-> Score: {result['plausibility_score']}, Band: {result['decision']['band']}")

    def test_03_dosage_weighting(self):
        """Test dosage-weighted scoring."""
        print("\n[Test 3] Testing Dosage Weighting...")
        smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
        # High dose Aspirin, Low dose Caffeine
        weights_levels = ["HIGH", "LOW"]
        result = self.explainer.explain(smiles, weights_levels=weights_levels)
        self.assertIn("dose_analysis", result)
        weights = result["dose_analysis"]["normalized_weights"]
        self.assertTrue(weights[smiles[0]] > weights[smiles[1]], "High dose should have higher weight")
        print("-> Dosage weighting confirmed.")

    def test_04_lab_readiness_overrides(self):
        """Test risk overrides (simulated high risk)."""
        print("\n[Test 4] Testing Lab-Readiness Overrides...")
        # Force a feedback override to test logic
        smiles = ["C1=CC=C(C=C1)C2=CC=CC=C2"] # Biphenyl
        self.feedback_manager.submit_feedback(smiles, "TOXIC", "Test Override")
        result = self.explainer.explain(smiles)
        self.assertEqual(result["decision"]["band"], "DISCARD")
        self.assertEqual(result["metrics"]["risk_penalty"], 1.0)
        print("-> Risk override confirmed.")

    def test_05_adme_novelty(self):
        """Test ADME placeholders and Novelty detection."""
        print("\n[Test 5] Testing ADME & Novelty...")
        # Novel C50 chain
        novel_smiles = ["CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)O"]
        result = self.explainer.explain(novel_smiles)
        self.assertTrue(result["novelty_report"]["novel_compounds"] > 0)
        self.assertIn("adme_simulation", result)
        print("-> Novelty and ADME confirmed.")

    def test_06_qsar_integration(self):
        """Test QSAR bioactivity estimation."""
        print("\n[Test 6] Testing QSAR Integration...")
        smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O"]
        # Run explain WITHOUT IC50 -> should trigger QSAR
        result = self.explainer.explain(smiles)
        dose_analysis = result["dose_analysis"]
        self.assertIn("activity_source", dose_analysis)
        source = dose_analysis["activity_source"][smiles[0]]
        print(f"-> Activity Source: {source}")
        self.assertTrue("QSAR Estimate" in source)

if __name__ == "__main__":
    unittest.main()
