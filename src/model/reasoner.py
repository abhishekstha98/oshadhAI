import torch
import torch.nn as nn

class CombinationReasoner(nn.Module):
    """
    Set-level reasoning module for multi-compound combinations.
    
    Purpose: Aggregate compound embeddings to detect:
    - Redundancy (overlapping target predictions)
    - Coverage (target diversity)
    - Risk patterns (conflict signals)
    
    CRITICAL: This module does NOT simulate mechanistic biology or synergy.
    It learns patterns from formulation co-occurrence data and ChEMBL-grounded embeddings.
    """
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Aggregator to collapse set to single vector
        # We can use sum (coverage-like) and max (strongest signal)
        self.output_proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, compound_embeddings, mask=None):
        # compound_embeddings: [Batch, MaxSetSize, embed_dim]
        # mask: [Batch, MaxSetSize] (True if padding)
        
        # 1. Intra-set interactions (Self-Attention)
        attn_out, _ = self.attn(compound_embeddings, compound_embeddings, compound_embeddings, key_padding_mask=mask)
        
        # Residual connection
        h = self.norm(compound_embeddings + attn_out)
        
        # 2. Aggregation
        # Handle masking for aggregation
        if mask is not None:
            # fill padding with 0 (for sum) or -inf (for max)
            h_masked_sum = h.clone()
            h_masked_sum[mask] = 0
            
            h_masked_max = h.clone()
            h_masked_max[mask] = -1e9
        else:
            h_masked_sum = h
            h_masked_max = h
            
        sum_pool = h_masked_sum.sum(dim=1)
        max_pool = h_masked_max.max(dim=1).values
        
        # Concatenate coverage (sum) and intensity (max)
        combined = torch.cat([sum_pool, max_pool], dim=1)
        
        # Project to single embedding
        formula_embedding = self.output_proj(combined)
        return formula_embedding
