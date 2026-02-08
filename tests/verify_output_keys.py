import sys
from pathlib import Path
import json

# Setup Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.explain import Explainer

def verify_output_keys():
    smiles = ["O=C1NC(=O)N(C=C1)[C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O", "c1[nH]c2c(n1)nc(nc2O)O"]
    
    print("Initializing Explainer...")
    explainer = Explainer()
    print("Running inference...")
    data = explainer.explain(smiles)
    
    forbidden_keys = [
        "uncertainty_explained",
        "novelty_report",
        "feedback_status"
    ]
    
    forbidden_nested = [
        ("metrics", "uncertainty"),
        ("decision", "percentile"),
        ("biological_context", "coverage_interpretation"),
        ("biological_risk_assessment", "interpretation")
    ]
    
    errors = []
    
    # Check top-level keys
    for key in forbidden_keys:
        if key in data:
            errors.append(f"Found forbidden top-level key: {key}")
            
    # Check nested keys
    for parent, key in forbidden_nested:
        if parent in data and key in data[parent]:
            errors.append(f"Found forbidden nested key: {parent}.{key}")
            
    if errors:
        print("FAIL: Verification failed with errors:")
        for e in errors:
            print(f"- {e}")
        sys.exit(1)
    else:
        print("PASS: All forbidden keys are absent.")
        # Print keys for confirmation
        print("\nExisting Top-Level Keys:")
        print(list(data.keys()))
        sys.exit(0)

if __name__ == "__main__":
    try:
        verify_output_keys()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
