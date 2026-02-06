import torch
import torch.nn as nn

class TargetPredictor(nn.Module):
    """
    Predicts compound-target interactions (multi-label classification).
    
    Supervision: ChEMBL experimental binding data ONLY.
    Purpose: Ground chemical structure in known biology.
    Does NOT: Predict novel interactions outside ChEMBL's experimental scope.
    """
    def __init__(self, embed_dim, num_targets):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, num_targets)
        )

    def forward(self, h):
        # h: [Batch, embed_dim]
        # returns logits: [Batch, num_targets]
        return self.net(h)

class ActivityRegressor(nn.Module):
    """
    Predicts pActivity (binding strength) values.
    
    Supervision: ChEMBL pChEMBL values (high-confidence assays only).
    Output: Predicted binding affinity (NOT efficacy or therapeutic effect).
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, h):
        return self.net(h)
