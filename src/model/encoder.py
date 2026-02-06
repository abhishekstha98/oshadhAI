import torch
import torch.nn as nn
import torch.nn.functional as F

class GINLayer(nn.Module):
    """
    Graph Isomorphism Network Layer.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
        self.eps = nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_attr=None):
        # x: [N, in_dim]
        # edge_index: [2, E]
        # Simple message passing: sum neighbors
        
        # 1. Message Passing (Scatter Add)
        row, col = edge_index
        out = torch.zeros_like(x, device=x.device)
        
        # Handling edge attributes if they matches in_dim (simplified)
        # For now, just aggregating neighbor features
        
        # Manual scatter add for robustness without torch_scatter
        out.index_add_(0, row, x[col])
        
        # 2. Aggregation with epsilon
        out = self.mlp((1 + self.eps) * x + out)
        return out

class MoleculeEncoder(nn.Module):
    """
    Encodes molecular graphs into fixed-size embeddings.
    """
    def __init__(self, atom_in_dim, hidden_dim=128, out_dim=256, num_layers=3):
        super().__init__()
        self.node_embedding = nn.Linear(atom_in_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            GINLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.final_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch_index=None):
        # x: [TotalNodes, atom_in_dim]
        # edge_index: [2, TotalEdges]
        # batch_index: [TotalNodes] identifying which graph the node belongs to
        
        h = self.node_embedding(x)
        
        for layer in self.layers:
            h = layer(h, edge_index)
            
        # Global Pooling (Sum)
        if batch_index is not None:
            # Global sum pool
            size = int(batch_index.max().item() + 1)
            out = torch.zeros(size, h.size(1), device=h.device)
            out.index_add_(0, batch_index, h)
        else:
            # Single graph case
            out = h.sum(dim=0, keepdim=True)
            
        return self.final_proj(out)
