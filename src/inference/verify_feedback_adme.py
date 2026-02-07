import sys
from pathlib import Path
import logging

# Adjust path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.inference.explain import Explainer
from src.inference.feedback import FeedbackManager

logging.basicConfig(level=logging.INFO)

def verify_feedback_adme():
    print("--- Starting Feedback & ADME Verification ---")
    
    explainer = Explainer()
    manager = FeedbackManager()
    
    # Test Case: Curcumin + Piperine
    smiles = [
        "COC1=CC(=CC(=C1)O)/C=C/C(=O)CC(=O)/C=C/C2=CC(=C(C=C2)O)OC", # Curcumin
        "C1CCN(CC1)C(=O)/C=C/C=C/C2=CC3=C(C=C2)OCO3" # Piperine
    ]
    
    # 1. Baseline Score
    print("\n1. Calculating Baseline...")
    base_res = explainer.explain(smiles)
    base_score = base_res["plausibility_score"]
    print(f"Baseline Score: {base_score}")
    print(f"ADME Data Present: {'adme_simulation' in base_res}")
    print(f"Novelty Report Present: {'novelty_report' in base_res}")
    
    if not base_res["novelty_report"]["known_compounds"] == 2:
        print("FAILURE: Validation compounds should be known.")
        
    # 2. VALID Feedback Test
    print("\n2. Testing VALID Feedback (+0.5 boost)...")
    manager.submit_feedback(smiles, "VALID", "Verified in lab")
    valid_res = explainer.explain(smiles)
    print(f"Score after VALID: {valid_res['plausibility_score']}")
    
    if valid_res['plausibility_score'] <= base_score:
        print("FAILURE: Score did not increase with VALID feedback.")
    else:
        print("SUCCESS: Score increased.")
        
    # 3. TOXIC Feedback Test
    print("\n3. Testing TOXIC Feedback (Risk Override)...")
    manager.submit_feedback(smiles, "TOXIC", "Failed hERG assay")
    toxic_res = explainer.explain(smiles)
    print(f"Risk Penalty: {toxic_res['metrics']['risk_penalty']}")
    print(f"Decision Band: {toxic_res['decision']['band']}")
    
    if toxic_res['metrics']['risk_penalty'] != 1.0:
        print("FAILURE: Risk penalty not maximized.")
    elif toxic_res['decision']['band'] != "DISCARD":
        print("FAILURE: Band not set to DISCARD.")
    else:
        print("SUCCESS: Risk override applied.")

    # 4. Novelty Test
    print("\n4. Testing Novel Compound Detection...")
    novel_smiles = [
        "C1=CC=C(C=C1)C2=CC=CC=C2" # Biphenyl (Likely known, but let's mix)
    ]
    # Completely made up SMILES (but valid) - extended to be sure
    novel_made_up = ["CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)O"] # Long fatty acid C50
    
    novel_res = explainer.explain(novel_made_up)
    print(f"Novelty Report: {novel_res['novelty_report']}")
    
    if novel_res['novelty_report']['novel_compounds'] > 0:
        print("SUCCESS: Novel compound detected.")
    else:
        print("WARNING: Test compound was found in database (unexpected).")

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    try:
        verify_feedback_adme()
    except Exception as e:
        print(f"CRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()
