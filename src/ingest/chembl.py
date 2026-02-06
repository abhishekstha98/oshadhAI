import duckdb
import logging
from pathlib import Path
from src.ingest.db import DB_PATH, init_db

# Constants for file names
FILE_COMPOUNDS = "chembl_compounds.csv"
FILE_TARGETS = "chembl_targets.csv"
FILE_ACTIVITY = "chembl_activity.csv"

def ingest_chembl_csv(data_dir: str, db_path: Path = DB_PATH):
    """
    Ingest ChEMBL data from CSV files into DuckDB.
    """
    root = Path(data_dir)
    p_compounds = root / FILE_COMPOUNDS
    p_targets = root / FILE_TARGETS
    p_activity = root / FILE_ACTIVITY
    
    # Check existence
    if not (p_compounds.exists() and p_targets.exists() and p_activity.exists()):
        logging.error(f"Missing ChEMBL CSVs in {data_dir}. Required: {FILE_COMPOUNDS}, {FILE_TARGETS}, {FILE_ACTIVITY}")
        return

    con = duckdb.connect(str(db_path))
    
    logging.info("Ingesting ChEMBL Targets...")
    # Observed columns: target_id, target_name, target_type
    con.execute(f"""
        INSERT OR IGNORE INTO targets (target_id, name, organism)
        SELECT 
            target_id as target_id, 
            target_name as name, 
            NULL as organism 
        FROM read_csv_auto('{str(p_targets)}', types={{'target_id': 'VARCHAR'}})
        WHERE target_type = 'SINGLE PROTEIN'
    """)
    
    logging.info("Ingesting ChEMBL Compounds...")
    # Observed columns: compound_id, smiles
    con.execute(f"""
        INSERT OR IGNORE INTO compounds (compound_id, smiles, inchi_key, source)
        SELECT 
            compound_id as compound_id, 
            smiles as smiles, 
            NULL as inchi_key, -- Missing in file, will need unification hash fallback
            'chembl' as source
        FROM read_csv_auto('{str(p_compounds)}', types={{'compound_id': 'VARCHAR'}})
        WHERE smiles IS NOT NULL
    """)

    logging.info("Ingesting ChEMBL Activities...")
    # Observed columns: compound_id, target_id, pActivity
    con.execute(f"""
        INSERT OR IGNORE INTO activities (compound_id, target_id, p_activity, assay_confidence, quality_weight)
        SELECT 
            compound_id as compound_id,
            target_id as target_id,
            pActivity as p_activity,
            9 as assay_confidence, -- Implied high confidence in filtered dataset
            1.0 as quality_weight
        FROM read_csv_auto('{str(p_activity)}', types={{'compound_id': 'VARCHAR', 'target_id': 'VARCHAR'}})
        WHERE pActivity IS NOT NULL
          AND target_id IN (SELECT target_id FROM targets)
          AND compound_id IN (SELECT compound_id FROM compounds)
    """)
    # Note on confidence: Detailed ChEMBL dumps have it. If simplified, we assume filtered.

    logging.info("ChEMBL CSV Ingestion Complete.")
    con.close()

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python -m src.ingest.chembl <datasets/ChEMBL>")
    else:
        init_db()
        ingest_chembl_csv(sys.argv[1])
