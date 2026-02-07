import os
import sys
import subprocess
import shutil
import pandas as pd
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "temp_data"
DB_PATH = DATA_DIR / "herb_combinator.duckdb"

def run_command(cmd, desc):
    print(f"\n--- {desc} ---")
    print(f"Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print("-> SUCCESS")
    except subprocess.CalledProcessError as e:
        print(f"-> FAILED with exit code {e.returncode}")
        sys.exit(1)

def setup_dummy_data():
    print("\n[Setup] Generatng dummy data artifacts...")
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir()
    
    # 1. ChEMBL Dummy Data
    chembl_dir = DATA_DIR / "ChEMBL"
    chembl_dir.mkdir()
    
    # compounds
    pd.DataFrame({
        'compound_id': ['CHEMBL1', 'CHEMBL2', 'CHEMBL3'],
        'smiles': ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'C1=CC=C(C=C1)C2=CC=CC=C2'],
        'pref_name': ['Aspirin', 'Caffeine', 'Biphenyl']
    }).to_csv(chembl_dir / "chembl_compounds.csv", index=False)
    
    # targets
    pd.DataFrame({
        'target_id': ['1', '2'],
        'target_name': ['COX-1', 'Adenosine A2A'], 
        'target_type': ['SINGLE PROTEIN', 'SINGLE PROTEIN']
    }).to_csv(chembl_dir / "chembl_targets.csv", index=False)
    
    pd.DataFrame({
        'compound_id': ['CHEMBL1', 'CHEMBL2', 'CHEMBL3'],
        'target_id': ['1', '2', '1'],
        'pActivity': [8.5, 7.2, 5.0],
        'standard_type': ['IC50', 'Ki', 'IC50']
    }).to_csv(chembl_dir / "chembl_activity.csv", index=False)
    
    # 2. HERB Dummy Data (Required for TCM linking)
    herb_dir = DATA_DIR / "HERB"
    herb_dir.mkdir()
    # We need HERB_ingredient_info_v2.csv to have 'Ingredient_name' and 'Canonical_smiles'
    # that matches what we put in TCM compounds.
    pd.DataFrame({
        'Ingredient_name': ['Aspirin', 'Caffeine'],
        'Canonical_smiles': ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'],
        'Ingredient_id': ['CHEMBL1', 'CHEMBL2'],
        'InChIKey': ['BSYNRYMUTXBXSQ-UHFFFAOYSA-N', 'RYYVLZVUVIJVGH-UHFFFAOYSA-N']
    }).to_csv(herb_dir / "HERB_ingredient_info_v2.csv", index=False)
    
    # HERB_herb_info_v2.csv (Just to pass file existence check)
    pd.DataFrame({
        'Herb_ID': ['HERB1'],
        'Herb_en_name': ['Willow']
    }).to_csv(herb_dir / "HERB_herb_info_v2.csv", index=False)
    
    # 3. TCM Dummy Data
    tcm_dir = DATA_DIR / "TCM"
    tcm_dir.mkdir()
    
    # medicinal_material.xlsx - Defines Herbs
    pd.DataFrame({
        'LATIN': ['Salix'],
        'CHINESE': ['Liu Shu'],
        'English Name': ['Willow Bark'],
        'Effects': ['Pain relief']
    }).to_excel(tcm_dir / "medicinal_material.xlsx", index=False)
    
    # medicinal_compound.xlsx - Links Herbs to Compound Names
    # ingest_tcm.py looks up these names in HERB_ingredient_info_v2.csv
    pd.DataFrame({
        'LATIN': ['Salix', 'Salix'],
        'COMPOUND': ['Aspirin', 'Caffeine'], # Caffeine in Willow? Just for testing.
        'ID': ['1', '2']
    }).to_excel(tcm_dir / "medicinal_compound.xlsx", index=False)

def main():
    print("=== STARTING FULL PIPELINE TEST ===")
    
    # 0. Setup
    setup_dummy_data()
    
    # Override DB Path env var for this process
    env = os.environ.copy()
    env["DB_PATH"] = str(DB_PATH)
    
    # We need to modify src/ingest/db.py to respect env var or pass it somehow?
    # Actually, src/ingest/db.py uses a hardcoded DB_PATH usually.
    # To avoid breaking prod DB, we will backup valid DB and restore it later.
    
    real_db = PROJECT_ROOT / "data" / "herb_combinator.duckdb"
    backup_db = PROJECT_ROOT / "data" / "herb_combinator.duckdb.bak"
    
    if real_db.exists():
        print(f"Backing up production DB to {backup_db}")
        shutil.move(str(real_db), str(backup_db))
    
    try:
        # 1. Ingest
        # We need to point ingestion to our dummy dirs
        run_command(f"python src/main.py ingest --chembl \"{DATA_DIR}/ChEMBL\" --herbs \"{DATA_DIR}/HERB\" --tcm \"{DATA_DIR}/TCM\"", "Step 1: Ingestion")
        
        # 2. Unify
        run_command("python src/main.py unify", "Step 2: Unification")
        
        # 3. Train Stage 1
        run_command("python src/main.py train-stage1 --epochs 1", "Step 3: Training Stage 1 (Encoder)")
        
        # 4. Train Stage 3
        run_command("python src/main.py train-stage3 --epochs 1", "Step 4: Training Stage 3 (Ranking)")
        
        # 5. Score
        run_command("python src/main.py score \"CC(=O)OC1=CC=CC=C1C(=O)O\"", "Step 5: Scoring (Inference)")
        
        print("\n=== FULL PIPELINE VERIFIED SUCCESSFULLY ===")
        
    finally:
        # Restore DB
        if backup_db.exists():
            print(f"Restoring production DB from {backup_db}")
            if real_db.exists():
                os.remove(real_db)
            shutil.move(str(backup_db), str(real_db))
            
        # Cleanup temp data
        if DATA_DIR.exists():
             shutil.rmtree(DATA_DIR)
             print("Cleaned up temp data.")

if __name__ == "__main__":
    main()
