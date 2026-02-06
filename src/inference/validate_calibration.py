import sys
import json
from pathlib import Path

# Adjust path to find src
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference.explain import Explainer

def run_sanity_checks():
    print("=== Antigravity Calibration Sanity Validator ===")
    explainer = Explainer()
    
    # Scenario 1: Trivial Solvent (should not rank high/investigate)
    print("\n[Scenario 1] Trivial Compound (Ethanol)")
    res1 = explainer.explain(["CCO"])
    
    # Assertion: If coverage is 0 (due to gating), it shouldn't be INVESTIGATE
    if res1['metrics']['coverage'] == 0:
        if res1['decision']['band'] not in ["DISCARD", "LOW_PRIORITY"]:
            print(f"FAIL: Trivial compound ranked as {res1['decision']['band']} despite zero gated coverage.")
            return False
    print(f"PASS: Trivial compound handled correctly (Band: {res1['decision']['band']})")

    # Scenario 2: High Risk Contradiction Test (Artificial stress)
    print("\n[Scenario 2] High Risk / High Uncertainty Contradiction")
    # We'll use a known complex molecule that might trigger uncertainty
    res2 = explainer.explain(["CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CCO"])
    
    # Assertion: If risk >= 0.85 and uncertainty >= 0.85, band MUST be DISCARD or LOW_PRIORITY
    r = res2['metrics']['risk_penalty']
    u = res2['metrics']['uncertainty']
    band = res2['decision']['band']
    
    print(f"Metrics - Risk: {r}, Uncertainty: {u}, Band: {band}")
    
    if r >= 0.85 and u >= 0.85:
        if band not in ["DISCARD", "LOW_PRIORITY"]:
            print(f"FAIL: Contradiction found. Risk {r} and Uncertainty {u} did not downgrade band {band}.")
            return False
            
    # Assertion: If risk >= 0.60, primary_risk_factors must not be empty
    if r >= 0.60 and not res2['toxicity_assessment']['primary_risk_factors']:
        print(f"FAIL: High risk ({r}) but primary_risk_factors is empty.")
        return False
        
    print("PASS: Risk/Uncertainty consistency rules verified.")

    # Scenario 3: Target Gating Assertion
    print("\n[Scenario 3] Target Gating Consistency")
    if res2['metrics']['coverage'] == 0 and "Broad biological engagement" in res2['biological_context']['coverage_interpretation']:
        print("FAIL: Coverage is 0 but interpretation claims broad engagement.")
        return False
    print("PASS: Target gating interpretation consistent.")

    print("\nALL SANITY CHECKS PASSED.")
    return True

if __name__ == "__main__":
    success = run_sanity_checks()
    if not success:
        sys.exit(1)
