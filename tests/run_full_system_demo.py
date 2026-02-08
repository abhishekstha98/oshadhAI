import sys
import os
import duckdb
import torch
import logging
import subprocess
import json
from pathlib import Path

# Setup Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingest.db import DB_PATH
from src.training.pretrain import train_stage1
from src.training.ranking import HerbDataset 

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def verify_ingestion():
    print("\n[PHASE 1] Ingestion Verification")
    if not os.path.exists(DB_PATH):
        print("FAIL: Database not found.")
        return False
    
    try:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        count = con.execute("SELECT COUNT(*) FROM compounds").fetchone()[0]
        con.close()
        print(f"PASS: Database loaded. Compound Count: {count}")
        return True
    except Exception as e:
        print(f"FAIL: Database error: {e}")
        return False

def run_training_smoke_test():
    print("\n[PHASE 2 & 3] Training Smoke Test")
    
    # Check if we can instantiate datasets (Phase 3)
    try:
        ds = HerbDataset()
        print(f"PASS: Stage 3 Dataset loaded. Herbs: {len(ds)}")
    except Exception as e:
        print(f"FAIL: Stage 3 Dataset Init Error: {e}")
        return False
        
    # We won't actually run full training to preserve the checkpoints/time
    # But we will check if the checkpoints exist
    ckpt_dir = project_root / "checkpoints"
    s1 = (ckpt_dir / "stage1.pt").exists()
    s3 = (ckpt_dir / "stage3.pt").exists()
    
    if s1 and s3:
        print("PASS: Checkpoints for Stage 1 and Stage 3 exist.")
        return True
    else:
        print(f"FAIL: Missing checkpoints. Stage1: {s1}, Stage3: {s3}")
        return False

def run_inference_cli(smiles):
    print("\n[PHASE 4] Inference Verification (CLI)")
    cmd = [sys.executable, str(project_root / "src" / "main.py"), "score", smiles]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print Stderr (logs)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
        
    # Print Stdout (JSON)
    print(result.stdout)
    
    try:
        # Validate JSON
        data = json.loads(result.stdout)
        if "plausibility_score" in data:
            return True
    except:
        pass
        
    return False

if __name__ == "__main__":
    print("=== OSHAD-AI FULL SYSTEM DEMO ===")
    
    ingest_ok = verify_ingestion()
    train_ok = run_training_smoke_test()
    
    smiles_input = "CC(=O)OC1=CC=CC=C1C(=O)O" # Aspirin
    infer_ok = run_inference_cli(smiles_input)
    
    if ingest_ok and train_ok and infer_ok:
        print("\n=== SUCCESS: All Phases Verified ===")
        sys.exit(0)
    else:
        print("\n=== FAILURE: Some phases failed ===")
        sys.exit(1)
