import duckdb
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ingest.db import DB_PATH

def debug():
    print("--- Database Counts ---")
    try:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        counts = [
            ('compounds', 'SELECT count(*) FROM compounds'),
            ('herbs', 'SELECT count(*) FROM herbs'),
            ('herb_compounds', 'SELECT count(*) FROM herb_compounds'),
            ('master_compounds', 'SELECT count(*) FROM master_compounds')
        ]
        for name, sql in counts:
            try:
                c = con.execute(sql).fetchone()[0]
                print(f"{name}: {c}")
            except Exception as e:
                print(f"{name}: Error {e}")
    except Exception as e:
        print(f"DB Error: {e}")

    print("\n--- TCM Excel Headers ---")
    try:
        p_comp = Path("datasets/TCM/medicinal_compound.xlsx")
        if p_comp.exists():
            df = pd.read_excel(p_comp, nrows=3)
            print(f"File: {p_comp}")
            print(f"Columns: {df.columns.tolist()}")
            print(df.to_string())
        else:
            print("TCM medicinal_compound.xlsx not found.")

        p_mat = Path("datasets/TCM/medicinal_material.xlsx")
        if p_mat.exists():
            df_mat = pd.read_excel(p_mat, nrows=3)
            print(f"\nFile: {p_mat}")
            print(f"Columns: {df_mat.columns.tolist()}")
            print(df_mat.to_string())
    except Exception as e:
        print(f"Excel Error: {e}")

if __name__ == "__main__":
    debug()
