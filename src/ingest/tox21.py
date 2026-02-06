import duckdb
import logging
from pathlib import Path
from rdkit import Chem
from src.ingest.db import DB_PATH

FILE_TOX21 = "tox21_10k_data_all.sdf"

def ingest_tox21_sdf(data_dir: str, db_path: Path = DB_PATH):
    """
    Ingest Tox21 SDF for risk flags.
    """
    root = Path(data_dir)
    p_sdf = root / FILE_TOX21
    
    if not p_sdf.exists():
        p_sdf = root / "Tox21" / FILE_TOX21
    
    if not p_sdf.exists():
        logging.error(f"Missing Tox21 SDF in {data_dir}. Required: {FILE_TOX21}")
        return

    con = duckdb.connect(str(db_path))
    
    # Create Risk Table if not exists
    con.execute("""
        CREATE TABLE IF NOT EXISTS risk_flags (
            compound_id VARCHAR,
            source VARCHAR,
            flag_name VARCHAR,
            score FLOAT,
            FOREIGN KEY (compound_id) REFERENCES compounds(compound_id)
        )
    """)
    
    logging.info("Reading Tox21 SDF (this may take a moment)...")
    suppl = Chem.SDMolSupplier(str(p_sdf))
    
    data_records = []
    
    for mol in suppl:
        if mol is None: continue
        
        try:
            # We need a compound ID. Tox21 usually has 'DSSTox_CID' or we use SMILES hash.
            # Ideally we match by InChIKey to existing ChEMBL/HERB compounds.
            # For now, we ingest the compound structure if new.
            
            props = mol.GetPropsAsDict()
            smiles = Chem.MolToSmiles(mol)
            
            # Use DSSTox_CID as primary if available, else skip or hash
            cid = props.get('DSSTox_CID', None)
            if not cid: continue
            
            # Insert Compound
            # We treat Tox21 as a valid structural source
            con.execute("INSERT OR IGNORE INTO compounds (compound_id, smiles, source) VALUES (?, ?, ?)", 
                        (str(cid), smiles, 'tox21'))
            
            # Extract Toxicity Flags (Active=1)
            # Tox21 props are like 'NR-AR', 'NR-AhR', etc. 0 or 1.
            for key, val in props.items():
                # Heuristic: keys that are usually targets
                if key.startswith('NR-') or key.startswith('SR-'):
                    if val == 1:
                        con.execute("INSERT INTO risk_flags VALUES (?, 'tox21', ?, 1.0)", (str(cid), key))
                        
        except Exception as e:
            continue

    logging.info("Tox21 Ingestion Complete.")
    con.close()

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python -m src.ingest.tox21 <datasets/>")
    else:
        ingest_tox21_sdf(sys.argv[1])
