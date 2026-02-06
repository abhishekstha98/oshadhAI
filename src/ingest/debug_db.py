import duckdb
from src.ingest.db import DB_PATH

def debug():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    
    tables = ['compounds', 'targets', 'activities', 'master_compounds']
    for t in tables:
        try:
            count = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            print(f"{t}: {count}")
            if count > 0:
                print(con.execute(f"SELECT * FROM {t} LIMIT 1").fetchall())
        except Exception as e:
            print(f"{t}: Error {e}")
            
    # Test Loader Query
    print("\n--- Testing Loader Query ---")
    query = """
        SELECT count(*)
        FROM activities a
        JOIN compounds c ON a.compound_id = c.compound_id
        LEFT JOIN master_compounds m ON c.inchi_key = m.inchi_key
        WHERE c.smiles IS NOT NULL OR m.canonical_smiles IS NOT NULL
    """
    print("Loader Count:", con.execute(query).fetchone()[0])

if __name__ == "__main__":
    debug()
