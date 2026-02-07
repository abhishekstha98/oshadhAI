import unittest
from src.inference.scoring import normalize_potency

class TestIC50(unittest.TestCase):
    def test_normalize_potency(self):
        # High IC50 = Low Potency, Low IC50 = High Potency
        ic50s = [10.0, 1.0]
        # potencies = [1/10, 1/1] = [0.1, 1.0]
        # normalized = [0.1/1.1, 1.0/1.1] = [0.0909..., 0.9090...]
        res = normalize_potency(ic50s)
        self.assertAlmostEqual(res[0], 0.1/1.1, places=5)
        self.assertAlmostEqual(res[1], 1.0/1.1, places=5)
        self.assertGreater(res[1], res[0])
        
        # Test with zero (should be handled safely)
        ic50s = [10.0, 0.0]
        res = normalize_potency(ic50s)
        self.assertGreater(res[1], res[0])
        
        # Test with missing values
        ic50s = [1.0, None]
        # potency = [1/1, 1/10] = [1.0, 0.1]
        res = normalize_potency(ic50s)
        self.assertAlmostEqual(res[0], 1.0/1.1, places=5)
        
        # Test all equal
        ic50s = [1.0, 1.0, 1.0]
        self.assertEqual(normalize_potency(ic50s), [1/3, 1/3, 1/3])

if __name__ == '__main__':
    unittest.main()
