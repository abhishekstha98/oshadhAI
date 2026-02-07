import unittest
from src.inference.scoring import normalize_dosage, map_categorical_dosage

class TestDosage(unittest.TestCase):
    def test_normalize_dosage(self):
        # Normal case
        weights = [0.5, 0.5]
        self.assertEqual(normalize_dosage(weights), [0.5, 0.5])
        
        # Scaling up
        weights = [1.0, 1.0]
        self.assertEqual(normalize_dosage(weights), [0.5, 0.5])
        
        # Scaling down
        weights = [0.1, 0.1]
        self.assertEqual(normalize_dosage(weights), [0.5, 0.5])
        
        # Uneven weights
        weights = [1, 2, 7]
        self.assertEqual(normalize_dosage(weights), [0.1, 0.2, 0.7])
        
        # All zero
        weights = [0, 0, 0]
        self.assertEqual(normalize_dosage(weights), [1/3, 1/3, 1/3])
        
        # Empty
        self.assertEqual(normalize_dosage([]), [])

    def test_map_categorical_dosage(self):
        # Standard mapping
        levels = ["LOW", "MEDIUM", "HIGH"]
        self.assertEqual(map_categorical_dosage(levels), [0.33, 0.66, 1.0])
        
        # Case insensitive
        levels = ["low", "Medium", "high"]
        self.assertEqual(map_categorical_dosage(levels), [0.33, 0.66, 1.0])
        
        # Unknown levels (should default to MEDIUM=0.66)
        levels = ["LOW", "UNKNOWN", "HIGH"]
        self.assertEqual(map_categorical_dosage(levels), [0.33, 0.66, 1.0])

    def test_apply_dose_weighting_logic(self):
        from src.inference.scoring import apply_dose_weighting
        import torch
        
        # Simple test for combined logic (W_i = norm(w_i * p_i))
        # Since apply_dose_weighting handles tensor multiplication, let's test the weight calculation part
        # by checking how it would normalize
        weights = [0.5, 0.5]
        potencies = [0.1, 0.9]
        # combined = [0.05, 0.45] -> normalized = [0.1, 0.9]
        
        # Create a mock embedding [1, 2, 1]
        emb = torch.ones((1, 2, 1))
        res = apply_dose_weighting(emb, weights, potencies, "cpu")
        
        self.assertAlmostEqual(res[0, 0, 0].item(), 0.1, places=5)
        self.assertAlmostEqual(res[0, 1, 0].item(), 0.9, places=5)

if __name__ == '__main__':
    unittest.main()
