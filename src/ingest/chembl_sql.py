import mysql.connector
import duckdb
import logging
from src.ingest.db import DB_PATH

# User provided config
DB_CONFIG = {
    "host": "localhost",
    "user": "abhi",
    "password": "Hello@123", # Ideally via env var, but using user provided for now
    "database": "chembl_36"
}

def get_conn():
    return mysql.connector.connect(**DB_CONFIG)

def ingest_chembl_sql(db_path=DB_PATH):
    """
    Ingest data directly from ChEMBL MySQL database.
    """
    logging.info("Connecting to MySQL ChEMBL database...")
    try:
        mysql_con = get_conn()
        # buffered=False (default) streams rows. buffered=True loads ALL rows to RAM (slow start).
        cursor = mysql_con.cursor(dictionary=True, buffered=False) 
    except Exception as e:
        logging.error(f"Failed to connect to MySQL: {e}")
        return

    con = duckdb.connect(str(db_path))

    # 1. Compounds
    # We need: chembl_id (compound_id), smiles, inchi_key
    logging.info("Querying ChEMBL Compounds (MySQL)...")
    query_comp = """
        SELECT 
            md.chembl_id as compound_id, 
            cs.canonical_smiles as smiles, 
            cs.standard_inchi_key as inchi_key
        FROM molecule_dictionary md
        JOIN compound_structures cs ON md.molregno = cs.molregno
        WHERE cs.canonical_smiles IS NOT NULL
    """
    # Fetch in chunks or stream? For simplicity in MVP, we fetch all into memory (warning: RAM)
    # Better: Use pandas generic read or chunking.
    # DuckDB can't query MySQL directly easily without extensions usually. 
    # We'll use cursor iterator.
    
    # Batch insert into DuckDB
    con.execute("BEGIN TRANSACTION")
    
    # 1. Compounds
    logging.info("Counting ChEMBL Compounds...")
    cursor.execute("SELECT count(*) as cnt FROM compound_structures WHERE canonical_smiles IS NOT NULL")
    total_compounds = cursor.fetchone()['cnt']
    logging.info(f"Found {total_compounds} compounds.")
    
    logging.info("Querying ChEMBL Compounds (MySQL)...")
    cursor.execute(query_comp)
    
    from tqdm import tqdm
    
    pbar = tqdm(total=total_compounds, desc="Ingesting Compounds")
    count = 0 
    BATCH_SIZE = 50000
    
    # Prepare Pre-statement
    # targets table
    con.execute("DELETE FROM targets WHERE source = 'chembl'") # Reset ChEMBL data
    con.execute("DELETE FROM compounds WHERE source = 'chembl'") 
    con.execute("DELETE FROM activities WHERE source = 'chembl'") 
    
    while True:
        rows = cursor.fetchmany(BATCH_SIZE)
        if not rows: break
        
        # rows is list of dicts
        # Transform to list of tuples for DuckDB
        data = [(r['compound_id'], r['smiles'], r['inchi_key'], 'chembl') for r in rows]
        
        con.executemany("INSERT OR IGNORE INTO compounds (compound_id, smiles, inchi_key, source) VALUES (?, ?, ?, ?)", data)
        count += len(rows)
        pbar.update(len(rows))
            
    pbar.close()
    
    # 2. Targets
    logging.info("Counting ChEMBL Targets...")
    cursor.execute("SELECT count(*) as cnt FROM target_dictionary WHERE target_type = 'SINGLE PROTEIN'")
    total_targets = cursor.fetchone()['cnt']
    logging.info(f"Found {total_targets} targets.")
    
    logging.info("Querying ChEMBL Targets (MySQL)...")
    cursor.execute(query_target)
    
    pbar = tqdm(total=total_targets, desc="Ingesting Targets")
    count = 0
    while True:
        rows = cursor.fetchmany(BATCH_SIZE)
        if not rows: break
        data = [(r['target_id'], r['name'], r['organism'], 'chembl') for r in rows]
        con.executemany("INSERT OR IGNORE INTO targets (target_id, name, organism, source) VALUES (?, ?, ?, ?)", data)
        count += len(rows)
        pbar.update(len(rows))
        
    pbar.close()

    # 3. Activities
    # High confidence (9), pChEMBL value exists
    logging.info("Counting ChEMBL Activities...")
    # Count Query (Approximate or Exact - Exact for tqdm)
    count_query_act = """
        SELECT count(*) as cnt
        FROM activities act
        JOIN assays ass ON act.assay_id = ass.assay_id
        JOIN target_dictionary td ON ass.tid = td.tid
        WHERE act.pchembl_value IS NOT NULL 
          AND ass.confidence_score >= 9
          AND td.target_type = 'SINGLE PROTEIN'
    """
    cursor.execute(count_query_act)
    total_acts = cursor.fetchone()['cnt']
    logging.info(f"Found {total_acts} activities.")

    logging.info("Querying ChEMBL Activities (MySQL)...")
    cursor.execute(query_act)
    
    pbar = tqdm(total=total_acts, desc="Ingesting Activities")
    count = 0
    while True:
        rows = cursor.fetchmany(BATCH_SIZE)
        if not rows: break
        
        # We assume standard weights for ChEMBL
        data = [(r['compound_id'], r['target_id'], float(r['pActivity']), 'chembl', 9, 1.0) for r in rows]
        
        # Activity table: compound_id, target_id, pActivity, source, assay_confidence, quality_weight
        con.executemany("""
            INSERT OR IGNORE INTO activities 
            (compound_id, target_id, pActivity, source, assay_confidence, quality_weight) 
            VALUES (?, ?, ?, ?, ?, ?)
        """, data)
        count += len(rows)
        pbar.update(len(rows))

    pbar.close()

    con.execute("COMMIT")
    con.close()
    mysql_con.close()
    logging.info("ChEMBL SQL Ingestion Complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_chembl_sql()
