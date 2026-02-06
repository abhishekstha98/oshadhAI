import duckdb
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ingest.db import DB_PATH

try:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    print("--- Compounds by Source ---")
    print(con.execute("SELECT source, count(*) FROM compounds GROUP BY source").fetchall())
    
    print("\n--- Activities by Source ---")
    print(con.execute("SELECT source, count(*) FROM activities GROUP BY source").fetchall())

    print("\n--- Herb Links (herb_compounds) ---")
    count_hc = con.execute("SELECT count(*) FROM herb_compounds").fetchone()[0]
    print(f"herb_compounds: {count_hc}")
    
except Exception as e:
    print(e)
