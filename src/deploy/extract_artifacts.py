import duckdb
import pandas as pd
import json
import os
import logging
from pathlib import Path
import sys

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "herb_combinator.duckdb"
OUT_DIR = PROJECT_ROOT / "data" / "inference"

def ensure_dir():
    if not OUT_DIR.exists():
        os.makedirs(OUT_DIR)
        logging.info(f"Created directory: {OUT_DIR}")

def extract_known_compounds():
    """Extract InChIKeys to Compressed JSON."""
    out_file = OUT_DIR / "known_compounds.json.gz"
    logging.info("Extracting Known Compounds...")
    
    try:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        # Fetch as list of tuples
        rows = con.execute("SELECT inchi_key FROM compounds").fetchall()
        con.close()
        
        keys = [r[0] for r in rows]
        logging.info(f"Loaded {len(keys)} compounds. Saving to JSON.GZ...")
        
        import gzip
        with gzip.open(out_file, 'wt', encoding='utf-8') as f:
            json.dump(keys, f)
        
        size_mb = out_file.stat().st_size / (1024 * 1024)
        logging.info(f"Saved {out_file} ({size_mb:.2f} MB)")
        
    except Exception as e:
        logging.error(f"Failed to extract compounds: {e}")
        sys.exit(1)

def extract_targets():
    """Extract Target ID mapping to JSON."""
    out_file = OUT_DIR / "targets.json"
    logging.info("Extracting Target Mappings...")
    
    try:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        
        # We need to reconstruct the EXACT mapping used in explain.py _load_target_names
        # 1. Get distinct target_ids from activities (the training set)
        targets_query = "SELECT DISTINCT target_id FROM activities"
        trained_target_ids = [t[0] for t in con.execute(targets_query).fetchall()]
        
        # 2. Get names
        names_query = "SELECT target_id, name FROM targets"
        all_names = con.execute(names_query).fetchall()
        id_to_name = {tid: name for tid, name in all_names}
        con.close()
        
        # 3. Build the final map: {index: name}
        # explain.py logic: return {i: id_to_name.get(tid[0], ...)}
        # We must preserve the order implied by 'enumerate(trained_targets)'
        # IMPORTANT: SQL query order is not guaranteed without ORDER BY. 
        # But explain.py didn't use ORDER BY. This is a potential flakiness in the original code.
        # However, DuckDB usually returns consistent order for same query if data hasn't changed.
        # To be safe, let's replicate the exact explain.py logic.
        
        # Wait, usually 'trained_targets' in explain.py comes from `con.execute(targets_query).fetchall()`.
        # We did exactly that above.
        
        final_map = {}
        for i, tid in enumerate(trained_target_ids):
             final_map[i] = id_to_name.get(tid, f"ID_{tid}")
             
        with open(out_file, 'w') as f:
            json.dump(final_map, f, indent=2)
            
        size_kb = out_file.stat().st_size / 1024
        logging.info(f"Saved {out_file} ({size_kb:.2f} KB) - {len(final_map)} targets")
        
    except Exception as e:
        logging.error(f"Failed to extract targets: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if not DB_PATH.exists():
        logging.error(f"Database not found at {DB_PATH}")
        sys.exit(1)
        
    ensure_dir()
    extract_known_compounds()
    extract_targets()
    logging.info("Extraction Complete.")
