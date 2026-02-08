# Tox21 Dataset Usage

## 1. Overview
**Purpose**: Source of toxicology data for Risk Assessment.
**Source**: Toxicology in the 21st Century (Tox21) Challenge.
**Format**: SDF (Structure-Data File).

## 2. Ingestion Phase
- **Script**: `src/ingest/tox21.py`
- **Tables Populated**:
    - `compounds`: Enriched with toxicity flags.
    - `compound_targets`: Stores interactions with nuclear receptors (e.g., AhR, ER-alpha) and stress response pathways.
- **Processing**:
    - Activity outcomes (0/1) are stored as risk signals.

## 3. Training Phase
- **Role**: Indirect influence via Risk Penalty.
- **Note**: Tox21 data is *not* currently used to train the main Plausibility Scorer directly, but is used to calibrate the Risk Penalty term in the `PlausibilityScorer` loss function (if enabled).

## 4. Inference Phase (Critical)
- **Role**: **Biological Risk Assessment**.
- **Process**:
    - When a user submits a compound, the system checks for:
        1. **Direct Match**: Is this specific compound known to be toxic in Tox21?
        2. **Structural Similarity**: Does the GNN embedding map close to known Tox21 actives?
    - **Output**:
        - Triggers `risk_penalty` logic.
        - Populates `primary_risk_factors` (e.g., "Hepatic metabolic burden", "Nuclear receptor interaction").
        - Can force a `decision.band` downgrade (e.g., "INVESTIGATE" or "DISCARD").
