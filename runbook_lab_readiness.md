# Antigravity Runbook: Dosage & Lab-Readiness

## 1. Concepts & Framing
- **No Toxicity Claims**: All risk is framed as "Biological Risk Signals" grounded in target biology.
- **Dosage as Weight**: Dosage is a relative weight [0,1], not a clinical unit.
- **Hard Gating**: Confidence threshold $\tau=0.20$ suppresses low-evidence targets.
- **Consistency**: High risk or high uncertainty will automatically downgrade the decision band to prevent overinterpretation.

## 2. Command Reference

### Scoring with Dosage
```powershell
# Weighted scoring (Numerical)
python src/main.py score "SMILES1" "SMILES2" --weights 0.8 0.2

# Weighted scoring (Categorical)
python src/main.py score "SMILES1" "SMILES2" --weights-levels HIGH LOW
```

### Calibration & Validation
```powershell
# 1. Regenerate reference distribution
python src/inference/calibrate.py

# 2. Run Sanity Check (Hard assertions for lab-readiness)
python src/inference/validate_calibration.py

# 3. Run Output Schema Validator
python src/main.py score "SMILES1" "SMILES2" > result.json
python src/inference/validate_output.py result.json
```

## 3. Interpreting Lab-Readiness Overrides
If you see `rule_trace` containing `UNCERTAINTY_OVERRIDE` or `RISK_OVERRIDE`, the system has detected high evidence sparsity or high risk and capped the decision band (e.g., from HIGH_PRIORITY to INVESTIGATE).
