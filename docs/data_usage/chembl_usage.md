# ChEMBL Dataset Usage

## 1. Overview
**Purpose**: Primary source for compound-target bioactivity data.
**Source**: ChEMBL Database (v33 or later).
**Format**: CSV extract (`chembl_extract.csv`).

## 2. Ingestion Phase
- **Script**: `src/ingest/chembl.py`
- **Tables Populated**:
    - `compounds`: Stores unique canonical SMILES and InChIKeys.
    - `targets`: Stores biological target definitions (Uniprot IDs, PrefNames).
    - `activities`: Stores bioactivity measures (IC50, Ki, EC50), normalized to pChEMBL values.
- **Normalization**:
    - Compounds are standardized to Canonical SMILES.
    - Activity values are converted to standard units (nM) and filtered for high-quality measurements.

## 3. Training Phase (Stage 1)
- **Role**: Pre-training the `MoleculeEncoder` and `TargetPredictor`.
- **Process**:
    - The model learns to predict biological targets from molecular structure.
    - **Input**: Molecular Graph (SMILES).
    - **Output**: Multi-label target probability vector.
    - **Loss**: Binary Cross Entropy (BCE) against known active targets.

## 4. Inference Phase
- **Role**:
    - **Target Prediction**: The pre-trained weights (`stage1.pt`) are used to predict likely targets for novel compounds.
    - **QSAR Proxies**: ChEMBL data informs the QSAR heuristics used for "estimated potency" when experimental IC50 is missing.
