import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model.encoder import MoleculeEncoder
from src.model.heads import TargetPredictor
from src.utils.loader import DrugTargetDataset, collate_fn # Stub collate
import logging
import os

def train_stage1(epochs=10, batch_size=32, lr=1e-3, save_path="checkpoints/stage1.pt"):
    """
    Stage 1: Train Compound -> Target Prediction.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # 1. Dataset
    dataset = DrugTargetDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # 2. Model
    # Dimensions match src/utils/graph.py atom_features
    # 43 (symbol) + 11 (degree) + 5 (Hs) + 6 (valence) + 1 (aromatic) = 66
    ATOM_DIM = 66
    
    encoder = MoleculeEncoder(atom_in_dim=ATOM_DIM).to(device)
    head = TargetPredictor(embed_dim=256, num_targets=dataset.num_targets).to(device)
    
    optimizer = optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=lr)
    criterion = nn.BCEWithLogitsLoss() # Multi-label
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    from tqdm import tqdm
    
    # 3. Loop
    encoder.train()
    head.train()
    
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        count = 0
        
        for batch in pbar:
            # batch conversion logic...
            # ... (No change to logic, just wrapped in pbar)
            
            # Re-implementing naive batching for the MVP custom graph object
            # Ideally use pygeometric.data.Batch.from_data_list
            
            # Simple approach: Loop (very slow) or simple concat
            # Let's try to concat node features and adjust edge indices
            
            batch_x = []
            batch_edge_index = []
            batch_batch_index = []
            batch_targets = []
            
            node_offset = 0
            
            for i, data in enumerate(batch):
                if data is None: continue 
                
                x = data['x'] 
                edge_index = data['edge_index']
                target_idx = data['target_idx'] 
                
                num_nodes = x.size(0)
                
                batch_x.append(x)
                batch_edge_index.append(edge_index + node_offset)
                batch_batch_index.append(torch.full((num_nodes,), i, dtype=torch.long))
                
                t_vec = torch.zeros(dataset.num_targets)
                if target_idx >= 0:
                    t_vec[target_idx] = 1.0
                batch_targets.append(t_vec)
                
                node_offset += num_nodes
            
            if not batch_x: continue
            
            x_cat = torch.cat(batch_x).to(device)
            edge_index_cat = torch.cat(batch_edge_index, dim=1).to(device)
            batch_index_cat = torch.cat(batch_batch_index).to(device)
            y_cat = torch.stack(batch_targets).to(device)
            
            optimizer.zero_grad()
            
            # Forward
            embeddings = encoder(x_cat, edge_index_cat, batch_index_cat)
            logits = head(embeddings)
            
            loss = criterion(logits, y_cat)
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            total_loss += current_loss
            count += 1
            
            pbar.set_postfix({'loss': f"{current_loss:.4f}", 'avg_loss': f"{total_loss/count:.4f}"})
            
        logging.info(f"Epoch {epoch+1}/{epochs} Avg Loss: {total_loss/len(loader):.4f}")
        
    # Save
    torch.save({
        'encoder': encoder.state_dict(),
        'target_head': head.state_dict()  # Renamed for clarity in inference
    }, save_path)
    logging.info(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_stage1(epochs=2) # Short run for test
