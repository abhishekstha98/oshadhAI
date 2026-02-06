import torch
import torch.nn as nn

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
        
        # Uncertainty estimator (simple variance proxy or direct regression)
        self.uncertainty_head = nn.Linear(embed_dim, 1)

    def forward(self, formula_embedding):
        # coverage: higher is better
        coverage = torch.sigmoid(self.coverage_head(formula_embedding))
        
        # penalties: constrained to be positive (0 to 1 range usually good for stability)
        redundancy = torch.sigmoid(self.redundancy_penalty_head(formula_embedding))
        risk = torch.sigmoid(self.risk_penalty_head(formula_embedding))
        
        uncertainty = F.softplus(self.uncertainty_head(formula_embedding))
        
        # Final Score Logic (Configurable)
        # Score = Coverage - (Redundancy * alpha) - (Risk * beta)
        # Here we output the components for the loss function to guide them
        # The final scalar might be assembled differently during inference
        
        # For training, we predict a "Plausibility Logit" or similar if we had labels.
        # Since this is "plausibility under known biology", we might train these heads 
        # via contrastive learning (positive sets vs negative sets).
        
        # Returning dictionary for modular loss calculation
        return {
            'coverage': coverage,
            'redundancy': redundancy,
            'risk': risk,
            'uncertainty': uncertainty
        }
import torch.nn.functional as F
