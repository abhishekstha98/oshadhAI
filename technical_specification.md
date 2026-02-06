# Antigravity: Master Technical Specification
**Version**: 1.1.0  
**Project Status**: Production Ready / Lab-Verified  
**Compliance Level**: Lab-Grade (Hard Confidence Gated)

---

## 1. System Overview
Antigravity is a high-throughput computational screener designed to rank the biological plausibility of multi-compound herbal formulations. It serves as a pre-experimental decision gate to reduce resource wastage in laboratory validation.

### 1.1 Core Value Proposition
- **Evidence-Grounded Ranking**: Scores are derived from 1.98M experimental compound-target activities (ChEMBL).
- **Dosage-Awareness**: Supports relative dosage weighting (LOW, MEDIUM, HIGH) to reflect realistic formulation balance.
- **Regulatory-Safe Output**: All terminology is reframed as "biological risk signals" to avoid medical overclaims.

---

## 2. Technical Architecture

### 2.1 Multi-Stage Pipeline
1.  **Stage 1: Compound Encoder (GNN)**
    - **Input**: SMILES (Graph-encoded).
    - **Task**: Multi-label target prediction (7,256 targets).
    - **Impact**: Learns biologically representative embeddings.
2.  **Stage 2: Combination Reasoner (Self-Attention)**
    - **Input**: Set of compound embeddings.
    - **Task**: Detect intra-set interactions (Redundancy & Multi-pathway coverage).
3.  **Stage 3: Plausibility Scorer**
    - **Input**: Formula embedding.
    - **Output**: 4 primary metrics (Coverage, Redundancy, Risk, Uncertainty).

### 2.2 Scoring Framework
Total Plausibility Score ($S_{total}$) is defined as:
$$S_{total} = w_{base} \times (C - R - S)$$
Where:
- **$C$ (Coverage)**: Breadth of biological engagement (targets met).
- **$R$ (Redundancy)**: Penalty for overlapping compound-target engagement.
- **$S$ (Risk)**: Aggregate biological risk signal (off-target activity).

---

## 3. Advanced Features

### 3.1 Dose-Weighted Integration
Dosage is modeled as a relative weight $w_i \in [0, 1]$. Compound embeddings are scaled prior to set interaction:
$$h_i' = w_i \times h_i$$
This ensures high-dose compounds dominate both coverage and risk signals.

### 3.2 Lab-Readiness Enforcement
To prevent "hallucinated" biological claims, the system applies hard consistency rules:
- **Confidence Gating ($\tau=0.20$)**: Targets below $0.20$ probability are suppressed.
- **Consistency Overrides**: If Risk Penalty $\ge 0.80$ or Uncertainty $\ge 0.85$, the decision band is automatically capped (e.g., to `INVESTIGATE` or `LOW_PRIORITY`).
- **Rule Tracing**: Every decision includes a `rule_trace` explaining exactly why a band was assigned or overridden.

---

## 4. Resource Components

### 4.1 Target Allowlists
The system uses curated files (`src/resources/risk_target_allowlists.json`) to map predicted targets to risk categories:
- **Cardiovascular**: hERG, KCNH2
- **Metabolic Burden**: CYP3A4, CYP2D6
- **Hepatic Burden**: BSEP, MRP2

---

## 5. Verification Framework

### 5.1 Stratified Calibration
Scores are calibrated against 1,000 stratified random formulations (sizes 2-5). This converts raw scores into percentiles (e.g., "Top 10%").

### 5.2 Assertions
Automated validators (`validate_calibration.py`) enforce the following:
- **Assertion A**: Zero-coverage formulations cannot rank above `LOW_PRIORITY`.
- **Assertion B**: High-risk formulations must provide traceable risk factors.
- **Assertion C**: Percentile rankings cannot contradict high risk/uncertainty signals.

---

## 6. Safety & Compliance
- **Non-Medical Framing**: Disclaimer included in every output.
- **Vocabulary Control**: Forbidden words (toxic, safe, dosage, synergy) are filtered programmatically.
- **Transparency**: Every score is accompanied by a breakdown of contributions and evidence density.
