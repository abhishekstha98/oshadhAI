import torch
import torch.nn as nn
import torch.nn.functional as F

class PlausibilityScorer(nn.Module):
    """
    Plausibility scoring head for multi-compound combinations.
    
    CRITICAL CONSTRAINTS (from authoritative knowledge base):
    - This module does NOT predict synergy, efficacy, or clinical outcomes
    - Output is for RANKING combinations by biological coherence only
    - Scores reflect plausibility under known compound-target data (ChEMBL)
    - Mandatory explanation required for all outputs
    
    Components:
    - Coverage: Target diversity (higher = broader biological engagement)
    - Redundancy Penalty: Overlapping target predictions (penalized)
    - Risk Penalty: Toxicity flags or metabolic conflict signals
    - Uncertainty: Evidence density proxy (high = low confidence)
    """
    def __init__(self, embed_dim):
        super().__init__()
        
        # Components of the score
        self.coverage_head = nn.Linear(embed_dim, 1)
        self.redundancy_penalty_head = nn.Linear(embed_dim, 1) # Should be positive, subtracted
        self.risk_penalty_head = nn.Linear(embed_dim, 1)       # Should be positive, subtracted
        
        # Uncertainty Decomposition Heads
        # 1. Data Sparsity: Evidence density proxy
        self.data_sparsity_head = nn.Linear(embed_dim, 1)
        # 2. Prediction Variance: Conflicting signal proxy
        self.prediction_variance_head = nn.Linear(embed_dim, 1)
        # 3. Out-of-Distribution: Chemical novelty proxy
        self.ood_head = nn.Linear(embed_dim, 1)

    def forward(self, formula_embedding):
        # coverage: higher is better
        coverage = torch.sigmoid(self.coverage_head(formula_embedding))
        
        # penalties: constrained to be positive
        redundancy = torch.sigmoid(self.redundancy_penalty_head(formula_embedding))
        risk = torch.sigmoid(self.risk_penalty_head(formula_embedding))
        
        # Uncertainty Components
        u_sparsity = F.softplus(self.data_sparsity_head(formula_embedding))
        u_variance = F.softplus(self.prediction_variance_head(formula_embedding))
        u_ood = F.softplus(self.ood_head(formula_embedding))
        
        # Total Uncertainty (Weighted Sum)
        uncertainty = u_sparsity + u_variance + u_ood
        
        return {
            'coverage': coverage,
            'redundancy': redundancy,
            'risk': risk,
            'uncertainty': uncertainty,
            'u_sparsity': u_sparsity,
            'u_variance': u_variance,
            'u_ood': u_ood
        }
