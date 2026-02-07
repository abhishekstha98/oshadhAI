import torch
import torch.nn.functional as F
import logging

def normalize_dosage(weights):
    """
    Normalize weights so that they sum to 1.0.
    Handle empty or all-zero weights by returning equal distribution.
    """
    if not weights:
        return []
    
    sum_w = sum(weights)
    if sum_w > 0:
        return [w / sum_w for w in weights]
    else:
        # Fallback to equal weights if sum is 0
        return [1.0 / len(weights)] * len(weights)

def normalize_potency(ic50_list):
    """
    Convert IC50 to potency (1/IC50) and normalize.
    Lower IC50 = Higher Potency.
    Handle missing/zero values gracefully by using a baseline potency.
    """
    if not ic50_list:
        return []
    
    # baseline IC50 for missing values (e.g. 10.0 uM)
    potencies = []
    for val in ic50_list:
        try:
            v = float(val) if val is not None else 10.0
            if v <= 0: v = 1e-6 # prevent div by zero
            potencies.append(1.0 / v)
        except (ValueError, TypeError):
            potencies.append(1.0 / 10.0) # baseline
            
    sum_p = sum(potencies)
    if sum_p > 0:
        return [p / sum_p for p in potencies]
    else:
        return [1.0 / len(ic50_list)] * len(ic50_list)

def map_categorical_dosage(levels):
    """
    Map categorical levels (LOW, MEDIUM, HIGH) to numerical weights.
    LOW = 0.33, MEDIUM = 0.66, HIGH = 1.0
    """
    mapping = {
        "LOW": 0.33,
        "MEDIUM": 0.66,
        "HIGH": 1.0
    }
    return [mapping.get(str(lv).upper(), 0.66) for lv in levels] # Default to MEDIUM if unknown

def apply_dose_weighting(embeddings, weights, potencies, device):
    """
    Apply combined dose/potency weighting to compound embeddings: h_i' = W_i * h_i
    W_i = normalize(w_i * p_i)
    
    Args:
        embeddings: Tensor of shape [Batch, MaxSetSize, embed_dim]
        weights: List of normalized weights
        potencies: List of normalized potencies
        device: Torch device
        
    Returns:
        Weighted embeddings tensor.
    """
    if not weights and not potencies:
        return embeddings
        
    # Use baseline of 1.0 if one is missing
    w_vec = weights if weights else [1.0] * len(potencies)
    p_vec = potencies if potencies else [1.0] * len(weights)
    
    # Combined weight
    combined = [w * p for w, p in zip(w_vec, p_vec)]
    norm_combined = normalize_dosage(combined)
    
    w_tensor = torch.tensor(norm_combined, device=device, dtype=torch.float).view(1, -1, 1)
    return embeddings * w_tensor

