# HERB Dataset Usage

## 1. Overview
**Purpose**: Source of high-quality Traditional Chinese Medicine (Generic) formulations and their constituent compounds.
**Source**: HERB Database (http://herb.ac.cn/).
**Format**: CSV files (`herb.csv`, `herb_compound.csv`).

## 2. Ingestion Phase
- **Script**: `src/ingest/herb.py`
- **Tables Populated**:
    - `herbs`: Stores herb metadata (Latin name, Chinese name).
    - `herb_compounds`: Link table mapping Herbs to Compounds.
    - `compounds`: Enriched with natural product compounds.
- **Unification**:
    - Compound names/IDs are unified against the Master Compound Table to prevent duplication with ChEMBL/TCM-Mesh.

## 3. Training Phase (Stage 3)
- **Role**: Training the Ranking Model (Plausibility Scorer).
- **Process**:
    - **Dataset**: `HerbDataset` treats known HERB formulations as "Positive Samples".
    - **Negative Sampling**: Random combinations of compounds are generated as "Negative Samples".
    - **Objective**: The model learns to rank user-submitted combinations higher than random noise based on the structural synergy patterns found in authentic HERB data.

## 4. Inference Phase
- **Role**:
    - **Validation**: HERB formulations serve as the ground truth for validating the plausibility score.
    - **Knowledge Base**: Used to flag if a user's input compound is a known constituent of famous herbs.
