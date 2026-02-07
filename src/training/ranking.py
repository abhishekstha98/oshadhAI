import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import duckdb
import random
from src.ingest.db import DB_PATH
from src.model.encoder import MoleculeEncoder
from src.model.reasoner import CombinationReasoner
from src.model.scorer import PlausibilityScorer
from src.utils.graph import smile_to_graph
import logging
import os

class HerbDataset(Dataset):
    def __init__(self, db_path=DB_PATH):
        self.con = duckdb.connect(str(db_path), read_only=True)
        # Load all herbs and their compounds
        self.herbs = {} # herb_id -> [smiles]
        
        # Load all herbs and their compounds
        # Join through master_compounds for best SMILES
        self.herbs = {} # herb_id -> [smiles]
        
        # We try to link herb_compounds -> compounds -> master_compounds
        rows = self.con.execute("""
            SELECT hc.herb_id, coalesce(m.canonical_smiles, c.smiles)
            FROM herb_compounds hc
            JOIN compounds c ON hc.compound_id = c.compound_id
            LEFT JOIN master_compounds m ON c.inchi_key = m.inchi_key
            WHERE c.smiles IS NOT NULL OR m.canonical_smiles IS NOT NULL
        """).fetchall()
        
        for hid, smile in rows:
            if not smile: continue
            if hid not in self.herbs:
                self.herbs[hid] = []
            self.herbs[hid].append(smile)
            
        self.herb_ids = list(self.herbs.keys())
        self.all_smiles = [s for sub in self.herbs.values() for s in sub]
        
    def __len__(self):
        return len(self.herb_ids)
        
    def __getitem__(self, idx):
        # Positive Sample
        hid = self.herb_ids[idx]
        pos_smiles = self.herbs[hid]
        
        # Negative Sample Strategy: Random Mix / Disjoint
        # Simple random sampling for MVP
        k = len(pos_smiles)
        neg_smiles = random.choices(self.all_smiles, k=k)
        
        return pos_smiles, neg_smiles

def process_batch_smiles(smiles_list_of_lists, encoder, device):
    """
    Convert a batch of sets of smiles to embeddings.
    smiles_list_of_lists: batch_size list of (list of smiles)
    Returns: [Batch, MaxSet, Embed], Mask
    """
    # 1. Flatten to process with encoder
    # This acts as a mini-collator inside the loop for simplicity via existing APIs
    # Ideal: pre-collate.
    
    # Check max len
    max_len = max(len(s) for s in smiles_list_of_lists)
    batch_size = len(smiles_list_of_lists)
    embed_dim = 256 # Match encoder output
    
    # Placeholder for output
    out = torch.zeros(batch_size, max_len, embed_dim, device=device)
    mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=device) # True = padding (torch convention differs for diff modules, MultiheadAttention uses True for ignore)
    
    # Process each molecule
    # Extremely slow to do one by one. Bulk process?
    # Bulk: Collect all unique unique smiles, encode, then scatter back.
    
    flat_smiles = [s for sub in smiles_list_of_lists for s in sub]
    # Unique to save encode time
    unique_smiles = list(set(flat_smiles))
    smile_to_idx = {s: i for i, s in enumerate(unique_smiles)}
    
    # Encode all unique
    # ... convert to graphs ...
    # This is heavy. For MVP training loop, let's mock the encoder call or do it simple.
    # We call smile_to_graph on all unique
    
    # Building batch for encoder
    batch_x, batch_edge_index, batch_batch_index = [], [], []
    valid_indices = []
    
    offset = 0
    valid_map = {} # unique_idx -> stored_idx
    
    for i, s in enumerate(unique_smiles):
        g = smile_to_graph(s)
        if g is None: continue
        
        x, edge_index, _ = g
        n = x.size(0)
        
        batch_x.append(x)
        batch_edge_index.append(edge_index + offset)
        batch_batch_index.append(torch.full((n,), len(valid_indices), dtype=torch.long))
        
        valid_map[i] = len(valid_indices)
        valid_indices.append(i)
        offset += n

    if not batch_x:
        return None, None

    x_cat = torch.cat(batch_x).to(device)
    ei_cat = torch.cat(batch_edge_index, dim=1).to(device)
    bi_cat = torch.cat(batch_batch_index).to(device)
    
    with torch.no_grad(): # Fixed encoder for stage 3 or fine-tune? "Phased" usually implies frozen or slow fine-tune.
        # Let's keep gradients if we want to tune encoder, but typically heavy.
        # User said "Stage 3... Train combination-level ranking". 
        # Typically we freeze encoder here for stability.
        enc_out = encoder(x_cat, ei_cat, bi_cat) # [NumUnique, Embed]
        
    # Scatter back to [Batch, MaxSet, Embed]
    for b, smiles_set in enumerate(smiles_list_of_lists):
        for t, smile in enumerate(smiles_set):
            u_idx = smile_to_idx[smile]
            if u_idx in valid_map:
                mapped_idx = valid_map[u_idx]
                out[b, t] = enc_out[mapped_idx]
                mask[b, t] = False # Not padding
                
    return out, mask

def train_stage3(stage1_ckpt="checkpoints/stage1.pt", epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Models
    ATOM_DIM = 66
    encoder = MoleculeEncoder(atom_in_dim=ATOM_DIM).to(device)
    
    # Load weights
    if stage1_ckpt and os.path.exists(stage1_ckpt):
        ckpt = torch.load(stage1_ckpt, map_location=device)
        encoder.load_state_dict(ckpt['encoder'])
        logging.info("Loaded encoder weights.")
    
    encoder.eval() # Freeze encoder
    
    reasoner = CombinationReasoner(embed_dim=256).to(device)
    scorer = PlausibilityScorer(embed_dim=256).to(device)
    
    optimizer = optim.Adam(list(reasoner.parameters()) + list(scorer.parameters()), lr=1e-4)
    criterion = nn.MarginRankingLoss(margin=1.0)
    
    dataset = HerbDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: x) # Custom collate list of tuples
    
    from tqdm import tqdm
    
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        count = 0
        
        for batch in pbar:
            # batch is list of (pos_smiles, neg_smiles)
            pos_batch = [item[0] for item in batch]
            neg_batch = [item[1] for item in batch]
            
            # Encode
            pos_emb, pos_mask = process_batch_smiles(pos_batch, encoder, device)
            neg_emb, neg_mask = process_batch_smiles(neg_batch, encoder, device)
            
            if pos_emb is None or neg_emb is None: continue
            
            # Reason
            pos_formula = reasoner(pos_emb, mask=pos_mask)
            neg_formula = reasoner(neg_emb, mask=neg_mask)
            
            # Score
            # Use sum of components or just raw implicitly via head? 
            # Prompt says "Collapses to one plausibility score".
            
            pos_outs = scorer(pos_formula)
            neg_outs = scorer(neg_formula)
            
            # Simple heuristic formula for the "Score"
            # We want to MAXIMIZE Pos and MINIMIZE Neg
            # Score = Coverage - Redundancy - Risk
            
            def calc_score(d):
                return d['coverage'] - d['redundancy'] - d['risk']
            
            pos_score = calc_score(pos_outs)
            neg_score = calc_score(neg_outs)
            
            # Target is 1 (pos > neg)
            target = torch.ones(pos_score.size(), device=device)
            
            # Ranking Loss (Plausibility)
            loss_plausibility = criterion(pos_score, neg_score, target)
            
            # Uncertainty Loss (Heuristic)
            # We want Neg Uncertainty > Pos Uncertainty (Random mixes are more uncertain)
            # So input1=neg_uncertainty, input2=pos_uncertainty, target=1
            pos_unc = pos_outs['uncertainty']
            neg_unc = neg_outs['uncertainty']
            loss_uncertainty = criterion(neg_unc, pos_unc, target)
            
            loss = loss_plausibility + 0.1 * loss_uncertainty
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            total_loss += current_loss
            count += 1
            
            pbar.set_postfix({'loss': f"{current_loss:.4f}", 'avg_loss': f"{total_loss/count:.4f}"})
            
        logging.info(f"Epoch {epoch+1}/{epochs} Avg Loss: {total_loss/len(loader):.4f}")

    # Save
    torch.save({
        'reasoner': reasoner.state_dict(),
        'scorer': scorer.state_dict()
    }, "checkpoints/stage3.pt")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import os
    train_stage3()
