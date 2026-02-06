import duckdb
import logging
from pathlib import Path
from src.ingest.db import DB_PATH

# Constants
FILE_HERB_INFO = "HERB_herb_info_v2.csv"
FILE_INGREDIENT = "HERB_ingredient_info_v2.csv"

def ingest_herb_csv(data_dir: str, db_path: Path = DB_PATH):
    """
    Ingest HERB dataset from CSVs.
    """
    root = Path(data_dir)
    p_info = root / FILE_HERB_INFO
    p_ingred = root / FILE_INGREDIENT
    
    if not (p_info.exists() and p_ingred.exists()):
        logging.error(f"Missing HERB CSVs in {data_dir}. Required: {FILE_HERB_INFO}, {FILE_INGREDIENT}")
        return

    con = duckdb.connect(str(db_path))
    
    logging.info("Ingesting HERB Info...")
    # Schema: herb_id, name, source='herb'
    # CSV cols: Herb_ID, Herb_Name, ...
    con.execute(f"""
        INSERT OR IGNORE INTO herbs (herb_id, name, source)
        SELECT 
            Herb_ID as herb_id, 
            Herb_en_name as name, 
            'herb' as source
        FROM read_csv_auto('{str(p_info)}')
    """)
    
    # For ingredients, we need to populate 'compounds' table first if they are new.
    # HERB provides Ingredient_ID, Compound_Name, SMILES, etc.
    logging.info("Ingesting HERB Compounds (Structure only)...")
    
    # We use a temp table to process SMILES
    con.execute(f"""
        CREATE TEMP TABLE raw_ingredients AS 
        SELECT * FROM read_csv_auto('{str(p_ingred)}')
    """)
    
    # Insert new compounds
    # Mappings: Ingredient_id -> compound_id, Canonical_smiles -> smiles, InChIKey -> inchi_key
    con.execute("""
        INSERT OR IGNORE INTO compounds (compound_id, smiles, inchi_key, source)
        SELECT 
            Ingredient_id as compound_id, 
            Canonical_smiles as smiles, 
            InChIKey as inchi_key,
            'herb' as source
        FROM raw_ingredients
        WHERE Canonical_smiles IS NOT NULL
    """)
    
    logging.info("Ingesting HERB-Compound Mappings...")
    # ERROR: HERB_ingredient_info_v2.csv in this dataset version lacks Herb_ID.
    # We cannot link them. Limiting to just Compounds and Herbs populations.
    # The linking will rely on TCM dataset or future updates.
    logging.warning("Skipping HERB mapping: 'Herb_ID' column missing in ingredient file.")
    
    con.execute("DROP TABLE raw_ingredients")
    logging.info("HERB Ingestion Complete.")
    con.close()

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python -m src.ingest.herb <datasets/HERB>")
    else:
        ingest_herb_csv(sys.argv[1])
