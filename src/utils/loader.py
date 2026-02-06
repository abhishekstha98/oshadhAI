import torch
from torch.utils.data import Dataset
import duckdb
from src.ingest.db import DB_PATH
from src.utils.graph import smile_to_graph
import logging

class DrugTargetDataset(Dataset):
    def __init__(self, db_path=DB_PATH, split='train'):
        self.con = duckdb.connect(str(db_path), read_only=True)
        
        # Simple split logic (in reality, use scaffold split)
        # Fetching all valid (compound, target, activity) triplets
        # This is a stub for the full logic.
        
        # For MVP: Load compounds that have activities
        # For MVP: Load compounds that have activities
        # We join with master_compounds to get the best canonical SMILES and Master ID
        query = """
            SELECT 
                coalesce(m.master_id, c.compound_id) as compound_id, 
                coalesce(m.canonical_smiles, c.smiles) as smiles, 
                a.target_id, 
                a.p_activity
            FROM activities a
            JOIN compounds c ON a.compound_id = c.compound_id
            LEFT JOIN master_compounds m ON c.inchi_key = m.inchi_key
            WHERE c.smiles IS NOT NULL OR m.canonical_smiles IS NOT NULL
            LIMIT 10000 
        """
        # LIMIT for MVP speed for now, user will need to remove it for full train
        
        self.data = self.con.execute(query).fetchall()
        
        # Build target map
        targets = self.con.execute("SELECT DISTINCT target_id FROM activities").fetchall()
        self.target_to_idx = {t[0]: i for i, t in enumerate(targets)}
        self.num_targets = len(targets)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        cid, smiles, tid, p_act = self.data[idx]
        
        # Graph
        graph = smile_to_graph(smiles)
        if graph is None:
            # Handle invalid smiles - should be filtered ideally
            return self.__getitem__((idx + 1) % len(self))
            
        x, edge_index, edge_attr = graph
        
        # Target
        y_idx = self.target_to_idx.get(tid, -1)
        
        return {
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'target_idx': torch.tensor(y_idx, dtype=torch.long),
            'p_activity': torch.tensor(p_act if p_act else 0.0, dtype=torch.float)
        }

def collate_fn(batch):
    # Custom collate to batch graphs
    # Assuming standard PyG formatting logic manual implementation for strictness
    # (Or use PyG DataLoader if available, but staying safe with torch)
    
    # Simple batching: list of graphs
    return batch
