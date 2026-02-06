import duckdb
import logging
from src.ingest.db import DB_PATH

def ingest_chembl_direct(db_path=DB_PATH):
    """
    Ingest data using DuckDB's native MySQL extension. 
    This bypasses Python object overhead and is orders of magnitude faster.
    """
    con = duckdb.connect(str(db_path))
    logging.info("Installing DuckDB MySQL extension...")
    try:
        con.execute("INSTALL mysql; LOAD mysql;")
    except Exception as e:
        logging.error(f"Failed to install/load MySQL extension: {e}")
        logging.error("Ensure you have internet access or the extension pre-installed.")
        return

    logging.info("Attaching to MySQL database...")
    # Note: This requires standard libmysqlclient to be available on the system
    try:
        con.execute("ATTACH 'host=localhost user=abhi password=Hello@123 database=chembl_36' AS mysqldb (TYPE MYSQL)")
    except Exception as e:
        logging.error(f"Failed to attach MySQL: {e}")
        return

    con.execute("BEGIN TRANSACTION")

    # 1. Compounds
    logging.info("Copying Compounds (Direct DB-to-DB)...")
    con.execute("DELETE FROM compounds WHERE source = 'chembl'")
    con.execute("""
        INSERT OR IGNORE INTO compounds (compound_id, smiles, inchi_key, source)
        SELECT 
            md.chembl_id, 
            cs.canonical_smiles, 
            cs.standard_inchi_key, 
            'chembl'
        FROM mysqldb.molecule_dictionary md
        JOIN mysqldb.compound_structures cs ON md.molregno = cs.molregno
        WHERE cs.canonical_smiles IS NOT NULL
    """)

    # 2. Targets
    logging.info("Copying Targets...")
    con.execute("DELETE FROM targets WHERE source = 'chembl'")
    con.execute("""
        INSERT OR IGNORE INTO targets (target_id, name, organism, source)
        SELECT 
            td.chembl_id, 
            td.pref_name, 
            td.organism, 
            'chembl'
        FROM mysqldb.target_dictionary td
        WHERE td.target_type = 'SINGLE PROTEIN'
    """)

    # 3. Activities
    logging.info("Copying Activities (High Confidence)...")
    con.execute("DELETE FROM activities WHERE source = 'chembl'")
    con.execute("""
        INSERT OR IGNORE INTO activities 
        (compound_id, target_id, p_activity, source, assay_confidence, quality_weight)
        SELECT 
            md.chembl_id, 
            td.chembl_id, 
            act.pchembl_value, 
            'chembl',
            9,
            1.0
        FROM mysqldb.activities act
        JOIN mysqldb.molecule_dictionary md ON act.molregno = md.molregno
        JOIN mysqldb.compound_structures cs ON md.molregno = cs.molregno
        JOIN mysqldb.assays ass ON act.assay_id = ass.assay_id
        JOIN mysqldb.target_dictionary td ON ass.tid = td.tid
        WHERE act.pchembl_value IS NOT NULL 
          AND ass.confidence_score >= 9
          AND td.target_type = 'SINGLE PROTEIN'
          AND cs.canonical_smiles IS NOT NULL
    """)

    con.execute("COMMIT")
    logging.info("Direct Ingestion Complete.")
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_chembl_direct()
