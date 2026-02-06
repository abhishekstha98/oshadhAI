import pandas as pd
from pathlib import Path

def inspect_csv(path):
    print(f"--- Inspecting {path.name} ---")
    try:
        df = pd.read_csv(path, nrows=2)
        print(df.columns.tolist())
    except Exception as e:
        print(f"Error: {e}")

def inspect_excel(path):
    print(f"--- Inspecting {path.name} ---")
    try:
        df = pd.read_excel(path, nrows=2)
        print(df.columns.tolist())
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    base = Path("datasets")
    
    inspect_csv(base / "ChEMBL/chembl_compounds.csv")
    inspect_csv(base / "HERB/HERB_ingredient_info_v2.csv")
    inspect_excel(base / "TCM/medicinal_compound.xlsx")
