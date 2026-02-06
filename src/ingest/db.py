import duckdb
import logging
from pathlib import Path

DB_PATH = Path("data/herb_combinator.duckdb")

def init_db(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    """
    Initialize the DuckDB database and create tables if they don't exist.
    """
    con = duckdb.connect(str(db_path))
    # Create tables IF NOT EXISTS
    # Compounds Table
    con.execute("""
        CREATE TABLE IF NOT EXISTS compounds (
            compound_id VARCHAR PRIMARY KEY,
            smiles VARCHAR NOT NULL,
            inchi_key VARCHAR,
            source VARCHAR
        )
    """)
    
    # Targets Table
    con.execute("""
        CREATE TABLE IF NOT EXISTS targets (
            target_id VARCHAR PRIMARY KEY,
            name VARCHAR,
            organism VARCHAR,
            source VARCHAR
        )
    """)
    
    # Activities Table
    con.execute("""
        CREATE TABLE IF NOT EXISTS activities (
            compound_id VARCHAR,
            target_id VARCHAR,
            p_activity FLOAT,
            source VARCHAR,
            assay_confidence INTEGER,
            quality_weight FLOAT,
            PRIMARY KEY (compound_id, target_id),
            FOREIGN KEY (compound_id) REFERENCES compounds(compound_id),
            FOREIGN KEY (target_id) REFERENCES targets(target_id)
        )
    """)
    
    # Herbs Table
    con.execute("""
        CREATE TABLE IF NOT EXISTS herbs (
            herb_id VARCHAR PRIMARY KEY,
            name VARCHAR,
            source VARCHAR
        )
    """)
    
    # Herb-Compound Mapping Table
    con.execute("""
        CREATE TABLE IF NOT EXISTS herb_compounds (
            herb_id VARCHAR,
            compound_id VARCHAR,
            evidence_weight FLOAT,
            PRIMARY KEY (herb_id, compound_id),
            FOREIGN KEY (herb_id) REFERENCES herbs(herb_id),
            FOREIGN KEY (compound_id) REFERENCES compounds(compound_id)
        )
    """)
    
    logging.info(f"Database initialized at {db_path}")
    return con

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
