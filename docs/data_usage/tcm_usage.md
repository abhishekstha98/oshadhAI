# TCM-Mesh / SymMap Dataset Usage

## 1. Overview
**Purpose**: Supplementary source of herb-compound-target relationships, providing broader coverage than HERB.
**Source**: TCM-Mesh / SymMap Databases.
**Format**: Excel/CSV files.

## 2. Ingestion Phase
- **Script**: `src/ingest/tcm.py`
- **Tables Populated**:
    - `herbs`: Supplementary herb entires.
    - `herb_compounds`: Additional mappings of compounds to herbs.
    - `targets`: Additional target interactions (if available).
- **Validation**:
    - Used to cross-validate compound occurrences found in HERB.

## 3. Training Phase (Stage 3)
- **Role**: Data Augmentation for Ranking Model.
- **Process**:
    - TCM-Mesh data is merged with HERB data to create a larger "Positive Sample" set for the Ranking Model.
    - Helps the model generalize to combinations not present in the stricter HERB database.

## 4. Inference Phase
- **Role**:
    - **Coverage**: Increases the likelihood that a user-submitted natural product is recognized by the system ("Known Compound").
    - **Novelty Detection**: Compounds not found in either HERB or TCM-Mesh are flagged as "Novel".
