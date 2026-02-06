"""
Validation script for improved Antigravity outputs.

Checks compliance with all 6 improvement requirements.
"""

import json
import logging

def validate_output(output):
    """Run all validation checks."""
    results = {
        "total_checks": 0,
        "passed": 0,
        "failed": 0,
        "details": []
    }
    
    def check(name, condition, message):
        results["total_checks"] += 1
        if condition:
            results["passed"] += 1
            results["details"].append(f"✅ {name}: {message}")
        else:
            results["failed"] += 1
            results["details"].append(f"❌ {name}: {message}")
    
    ## C1: Calibration & Decision Sanity Checks
    if "decision" in output:
        dec = output["decision"]
        metrics = output.get("metrics", {})
        valid_bands = ["DISCARD", "LOW_PRIORITY", "INVESTIGATE", "HIGH_PRIORITY", "CRITICAL"]
        
        check("DEC-1", dec.get("band") in valid_bands, f"Decision band valid: {dec.get('band')}")
        check("DEC-2", "rule_trace" in dec and len(dec["rule_trace"]) > 0, "Decision has rule_trace")
        
        # Hard Assertions for Lab-Readiness
        r = metrics.get("risk_penalty", 0)
        u = metrics.get("uncertainty", 0)
        c = metrics.get("coverage", 0)
        band = dec.get("band")
        
        if u >= 0.85:
            check("DEC-OVERRIDE-U", band not in ["HIGH_PRIORITY", "CRITICAL"], "Uncertainty override applied")
        if r >= 0.80:
            check("DEC-OVERRIDE-R", band not in ["HIGH_PRIORITY", "CRITICAL"], "Risk override applied")
        if c <= 0.02:
            check("DEC-OVERRIDE-C", band in ["DISCARD", "LOW_PRIORITY"], "Coverage (Gated) override applied")
    
    ## C2: Gating & Confidence Checks
    if "target_confidence_summary" in output:
        conf = output["target_confidence_summary"]
        check("CONF-1", conf.get("threshold") == 0.20, f"Threshold is 0.20: {conf.get('threshold')}")
        check("CONF-2", all(v >= 0 for v in conf.get("gated_target_counts", {}).values()), "Gated counts non-negative")

    ## C3: Biological Context Checks
    if "biological_context" in output:
        bio = output["biological_context"]
        metrics = output.get("metrics", {})
        check("BIO-1", isinstance(bio.get("predicted_targets"), list), f"Targets list: {len(bio.get('predicted_targets', []))}")
        
        # Assertion: If coverage is 0, interpretation must be negative
        if metrics.get("coverage", 0) == 0:
            check("BIO-GATED", "No high-confidence" in bio.get("coverage_interpretation", ""), "Coverage interpretation gating")
    
    ## C4: Risk Signal Checks
    if "biological_risk_assessment" in output:
        tox = output["biological_risk_assessment"]
        metrics = output.get("metrics", {})
        risk = metrics.get("risk_penalty", 0)
        
        if risk >= 0.60:
            check("RISK-FACTORS", len(tox.get("primary_risk_factors", [])) > 0, "Risk factors non-empty for high risk")
        
        check("RISK-LEVEL", tox.get("risk_level") in ["LOW", "MODERATE", "HIGH"], f"Risk level valid: {tox.get('risk_level')}")

    ## C5: Language Safety Checks
    disclaimer = output.get("disclaimer", "")
    check("SAFE-1", len(disclaimer) > 0, "Disclaimer present")
    
    forbidden_words = ["toxic", "safe", "lethal", "dosage", "fda", "clinical", "synergy", "efficacy", "cure", "treat"]
    output_str = json.dumps(output).lower()
    
    # Check for forbidden words while allowing "dose" in technical keys
    found_words = []
    for word in forbidden_words:
        if word in output_str:
            # Exception: 'dose' is allowed in keys, but 'dosage' is not allowed anywhere.
            # If word is 'safe', check if it's literally the word, not 'safety'? 
            # User said "safe" is forbidden.
            found_words.append(word)
            
    check("SAFE-3", len(found_words) == 0, f"Forbidden words found: {found_words}")
    
    ## C6: Counterfactual Checks
    if "suggestions" in output:
        suggs = output["suggestions"]
        check("CF-1", isinstance(suggs, list) and len(suggs) > 0,
              f"Suggestions present: {len(suggs)} items")
        
        conservative_words = ["consider", "may", "might", "could"]
        if suggs:
            first_sugg = suggs[0].lower()
            check("CF-2", any(word in first_sugg for word in conservative_words),
                  "Suggestions use conservative phrasing")
    
    # Summary
    results["pass_rate"] = results["passed"] / results["total_checks"] * 100
    
    return results

def print_validation_report(results):
    """Print formatted validation report."""
    print("=" * 60)
    print("ANTIGRAVITY OUTPUT VALIDATION REPORT")
    print("=" * 60)
    print(f"\nTotal Checks: {results['total_checks']}")
    print(f"Passed: {results['passed']} ({results['pass_rate']:.1f}%)")
    print(f"Failed: {results['failed']}")
    print("\nDetails:")
    print("-" * 60)
    for detail in results["details"]:
        print(detail)
    print("=" * 60)
    
    if results["pass_rate"] == 100:
        print("✅ ALL CHECKS PASSED")
    elif results["pass_rate"] >= 80:
        print("⚠️  MOSTLY PASSING - Review failures")
    else:
        print("❌ CRITICAL FAILURES - Do not deploy")

def main():
    """Run validation on test output."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python validate_output.py <output.json>")
        sys.exit(1)
    
    with open(sys.argv[1], 'r', encoding='utf-8-sig') as f:
        content = f.read().strip()
        # Find start of JSON if there's logging garbage
        start_idx = content.find('{')
        if start_idx != -1:
            content = content[start_idx:]
        output = json.loads(content)
    
    results = validate_output(output)
    print_validation_report(results)

if __name__ == "__main__":
    main()
