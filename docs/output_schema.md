# API Output Schema Reference

This document explains the JSON response structure returned by the `score` endpoint and CLI.

## Root Fields

### `plausibility_score` (float)
**Range**: `-1.0` to `1.0`
-   **Definition**: The primary ranking score indicating the biological feasibility of the formulation.
-   **Interpretation**:
    -   `> 0.5`: High plausibility (synergistic potential).
    -   `0.0 - 0.5`: Moderate plausibility.
    -   `< 0.0`: Low plausibility (likely inert or antagonistic).
-   **Note**: This is a *comparative* score trained against known effective formulations, not a probability of cure.

### `metrics` (Object)
Breakdown of the components feeding into the Plausibility Score.
-   **`coverage`** (0.0 - 1.0): How well the formulation covers the target biological network. Higher is better.
-   **`redundancy_penalty`** (0.0 - 1.0): Penalty for adding compounds that hit the same targets as existing ones (diminishing returns). Lower is better.
-   **`risk_penalty`** (0.0 - 1.0): Penalty derived from toxicity signals (Tox21/Allowlists). Lower is better.
-   **`uncertainty`** (0.0 - 1.0): Model's epistemic uncertainty. Higher values dampen the final score.

### `target_confidence_summary` (Object)
Details on the confidence of biological target predictions.
-   **`threshold`**: The minimum probability (e.g., 0.2) required to consider a target "active".
-   **`max_prob_per_compound`**: The highest single target probability found for each input compound. Useful for identifying "weak" contributors.
-   **`gated_target_counts`**: Number of targets per compound that passed the threshold.

### `decision` (Object)
Actionable recommendation based on the score and calibration.
-   **`band`**: Classification category (`CRITICAL`, `HIGH_PRIORITY`, `INVESTIGATE`, `LOW_PRIORITY`, `DISCARD`).
-   **`percentile`**: Where this formulation ranks compared to the validation dataset (e.g., 84.9 means it scores better than 84.9% of known data).
-   **`recommendation`**: Human-readable advice (e.g., "Prioritize for testing").
-   **`rule_trace`**: List of logic rules applied (e.g., "Downgraded due to High Risk").

### `dose_analysis` (Object)
Impact of dosage and potency on the scoring.
-   **`normalized_weights`**: Input dosage weights normalized to sum to 1.0.
-   **`normalized_potencies`**: Biological potency (from IC50 or QSAR) normalized to 0.0-1.0.
-   **`activity_source`**: Source of the potency data (e.g., "Experimental IC50" vs "QSAR Estimate").
-   **`combined_influence`**: The final scalar used to weight the compound's contribution in the graph embedding (`Weight * Potency`).

### `biological_risk_assessment` (Object)
Safety analysis based on Tox21 and structural alerts.
-   **`risk_penalty`**: The numeric penalty applied to the score.
-   **`risk_level`**: Categorical risk (`LOW`, `MODERATE`, `HIGH`).
-   **`primary_risk_factors`**: List of specific concerns (e.g., "Hepatic metabolic burden").
-   **`compound_level_flags`**: Specific risk flags mapping to each input compound.

### `contributors` (Array)
Contribution analysis for each compound in the mix.
-   **`contribution`**: The marginal impact of this compound on the total score. (Positive = adds value, Negative = hurts formulation).
-   **`top_targets`**: The most likely biological targets for this specific compound.

### `biological_context` (Object)
System-level biological interpretation.
-   **`predicted_targets`**: The top targets for the *entire formulation* (aggregated).
-   **`coverage_interpretation`**: Text summary of network engagement breadth.

### `adme_simulation` (Array)
Simulated pharmacokinetic properties (In-Silico).
-   **`logP_proxy`**: Estimated lipophilicity.
-   **`solubility_class`**: Predicted solubility (`High`, `Moderate`, `Low`).
-   **`blood_brain_barrier`**: predicted CNS penetration (`Permeable`, `Impermeable`).
-   **`metabolic_stability`**: (Placeholder) Future extension for half-life prediction.

### `novelty_report` (Object)
-   **`known_compounds`**: Count of input compounds present in the training DB.
-   **`novel_compounds`**: Count of input compounds *never seen before*.
-   **`inference_method`**: "Direct Lookup" (if known) or "Graph Isomorphism Network (Generalization)" (if novel).

### `disclaimer` (String)
**Mandatory**: Legal and scientific disclaimer stating that results are computational predictions and require experimental validation.
