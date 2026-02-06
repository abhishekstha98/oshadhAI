import pandas as pd
import duckdb
import logging
import os
from pathlib import Path
from src.ingest.db import DB_PATH

# Constants
FILE_MEDICINAL_MATERIAL = "medicinal_material.xlsx"
FILE_MEDICINAL_COMPOUND = "medicinal_compound.xlsx"
# FILE_PRESCRIPTION = "prescription.xlsx" # Not strictly required for MVP structural ingest

def ingest_tcm_excel(data_dir: str, db_path: Path = DB_PATH):
    """
    Ingest TCM dataset from Excel files.
    """
    root = Path(data_dir)
    p_material = root / FILE_MEDICINAL_MATERIAL
    p_compound = root / FILE_MEDICINAL_COMPOUND
    
    if not (p_material.exists() and p_compound.exists()):
        logging.error(f"Missing TCM Excel files in {data_dir}. Required: {FILE_MEDICINAL_MATERIAL}, {FILE_MEDICINAL_COMPOUND}")
        return

    con = duckdb.connect(str(db_path))
    
    logging.info("Reading TCM Medicinal Materials...")
    try:
        df_mat = pd.read_excel(p_material)
        df_comp = pd.read_excel(p_compound)
    except Exception as e:
        logging.error(f"Failed to read Excel files: {e}")
        return

    logging.info("Ingesting TCM Materials (Herbs)...")
    # Clean and rename for DuckDB
    # ID -> herb_id, Name -> name
    # Observed columns in Material: LATIN, COMMON, KOREAN...
    # Observed columns in Compound: LATIN, ID, COMPOUND...
    # The numeric 'ID' in Compound file is NOT in Material file.
    # We must use 'LATIN' as the shared key (herb_id).
    
    if 'LATIN' in df_mat.columns:
        df_mat_clean = df_mat.copy()
        # Use LATIN as ID (unique enough for this dataset)
        df_mat_clean['herb_id'] = df_mat_clean['LATIN'].astype(str).str.strip()
        
        # Name = LATIN + (CHINESE)
        if 'CHINESE' in df_mat.columns:
            df_mat_clean['name'] = df_mat_clean['LATIN'].fillna('') + ' (' + df_mat_clean['CHINESE'].fillna('') + ')'
        else:
             df_mat_clean['name'] = df_mat_clean['LATIN']
             
        df_mat_clean = df_mat_clean[['herb_id', 'name']]
        df_mat_clean['source'] = 'tcm'
        
        con.execute("CREATE TEMP TABLE temp_tcm_herbs AS SELECT * FROM df_mat_clean")
        con.execute("""
            INSERT OR IGNORE INTO herbs (herb_id, name, source)
            SELECT herb_id, name, source FROM temp_tcm_herbs
        """)
        con.execute("DROP TABLE temp_tcm_herbs")
    else:
        logging.warning(f"TCM Material columns mismatch. Found: {df_mat.columns}")

    logging.info("Ingesting TCM Compounds...")
    # TCM 'COMPOUND' column is Name. We need SMILES.
    # We try to match against HERB dataset if available.
    
    herb_ing_file = root.parent / "HERB" / "HERB_ingredient_info_v2.csv"
    name_to_smiles = {}
    
    if herb_ing_file.exists():
        logging.info(f"Found HERB ingredients at {herb_ing_file}. Building Name->SMILES map...")
        try:
            # Load only needed cols
            df_herb = pd.read_csv(herb_ing_file, usecols=['Ingredient_name', 'Canonical_smiles'])
            # Normalize names: lowercase, strip
            df_herb['clean_name'] = df_herb['Ingredient_name'].astype(str).str.lower().str.strip()
            df_herb = df_herb.dropna(subset=['Canonical_smiles'])
            # Create map
            name_to_smiles = pd.Series(df_herb.Canonical_smiles.values, index=df_herb.clean_name).to_dict()
            logging.info(f"Loaded {len(name_to_smiles)} ingredient structures from HERB.")
        except Exception as e:
            logging.warning(f"Failed to load HERB ingredients: {e}")
            
    # Map 'ID' -> compound_id, 'SMILES' -> smiles
    # In medicinal_compound.xlsx: 'LATIN', 'ID', 'COMPOUND'
    # 'LATIN' matches the new herb_id. 'COMPOUND' is Name.
    
    if 'COMPOUND' in df_comp.columns and 'LATIN' in df_comp.columns:
        # We need to resolve SMILES for these names
        merged_data = []
        
        for idx, row in df_comp.iterrows():
            c_name = str(row['COMPOUND']).strip()
            h_id = str(row['LATIN']).strip() # Herb ID is now LATIN name
            
            # Lookup SMILES
            smiles = name_to_smiles.get(c_name.lower())
            
            if smiles:
                # We have a structure!
                # Generate a hash ID for the compound or use name-hash
                import hashlib
                c_id = "TCM_" + hashlib.md5(smiles.encode()).hexdigest()[:10]
                
                merged_data.append({
                    'herb_id': h_id,
                    'compound_id': c_id,
                    'smiles': smiles,
                    'source': 'tcm_linked'
                })
        
        if merged_data:
            df_linked = pd.DataFrame(merged_data)
            logging.info(f"Resolved {len(df_linked)} TCM compounds to structures via HERB.")
            
            # 1. Insert Compounds
            con.execute("CREATE TEMP TABLE temp_linked_comp AS SELECT DISTINCT compound_id, smiles, source FROM df_linked")
            con.execute("""
                INSERT OR IGNORE INTO compounds (compound_id, smiles, source)
                SELECT compound_id, smiles, source FROM temp_linked_comp
            """)
            con.execute("DROP TABLE temp_linked_comp")
            
            # 2. Insert Links
            con.execute("CREATE TEMP TABLE temp_linked_hc AS SELECT herb_id, compound_id FROM df_linked")
            con.execute("""
                INSERT OR IGNORE INTO herb_compounds (herb_id, compound_id, evidence_weight)
                SELECT herb_id, compound_id, 0.5 FROM temp_linked_hc
            """)
            con.execute("DROP TABLE temp_linked_hc")
            
        else:
             logging.warning("No TCM compounds could be resolved to SMILES. herb_compounds will be empty.")
        
    logging.info("TCM Ingestion Complete.")
    con.close()

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python -m src.ingest.tcm <datasets/TCM>")
    else:
        ingest_tcm_excel(sys.argv[1])
