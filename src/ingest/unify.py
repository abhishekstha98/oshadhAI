import duckdb
import logging
from src.ingest.db import DB_PATH
from src.utils.chem import standardize_compound

def unify_compounds(db_path=DB_PATH):
    """
    Unify compounds from all sources into a master table.
    Resolution: InChIKey collision -> Prefer ChEMBL > Tox21 > HERB > TCM
    """
    con = duckdb.connect(str(db_path))
    
    # 1. Canonicalize all (in memory UDF or update rows)
    # Using python UDF on the 'compounds' table is slow for large datasets.
    # For MVP, we assume ingestion scripts did their best, but we do a pass here if needed.
    # We rely on InChIKeys being present.
    
    # 2. Master Table Creation
    # We use coalesce(inchi_key, md5(smiles)) as the grouping key since some datasets lack InChIKey
    con.execute("DROP TABLE IF EXISTS master_compounds")
    con.execute("""
        CREATE TABLE master_compounds (
            master_id VARCHAR PRIMARY KEY,
            inchi_key VARCHAR,
            canonical_smiles VARCHAR,
            primary_source VARCHAR,
            sources_list VARCHAR[],
            chembl_id VARCHAR,
            herb_id VARCHAR,
            tox21_id VARCHAR
        )
    """)
    
    query = """
    WITH scored AS (
        SELECT 
            compound_id, 
            smiles, 
            CASE WHEN inchi_key IS NULL OR inchi_key = '' THEN md5(smiles) ELSE inchi_key END as union_key, 
            source,
            CASE 
                WHEN source = 'chembl' THEN 1 
                WHEN source = 'tox21' THEN 2 
                WHEN source = 'herb' THEN 3
                ELSE 4 
            END as trust_rank
        FROM compounds
        WHERE smiles IS NOT NULL
    ),
    Unified AS (
        SELECT 
            union_key,
            list(source) as sources,
            arg_min(compound_id, trust_rank) as primary_id,
            arg_min(source, trust_rank) as primary_source,
            arg_min(smiles, trust_rank) as best_smiles,
            first(union_key) as f_key
        FROM scored
        GROUP BY union_key
    )
    INSERT OR IGNORE INTO master_compounds 
    SELECT 
        -- Generate internal source-agnostic ID
        'AG_' || f_key as master_id,
        CASE WHEN f_key LIKE 'InChI%' THEN f_key ELSE NULL END as inchi_key,
        best_smiles as canonical_smiles,
        primary_source,
        sources as sources_list,
        -- We can try to extract specific IDs if they exist in the group, 
        -- but for MVP we log the primary. 
        -- Ideally we would join back to get Chembl ID specifically if it exists in the cluster.
        CASE WHEN list_contains(sources, 'chembl') THEN primary_id ELSE NULL END as chembl_id,
        NULL as herb_id,
        NULL as tox21_id
    FROM Unified
    """
    con.execute(query)
    
    # Update cross-refs
    # Update chembl_id if present in clusters
    # (Simplified for MVP: Master ID is the primary ID)
    
    count = con.execute("SELECT count(*) FROM master_compounds").fetchone()[0]
    logging.info(f"Unified into {count} unique master compounds.")
    con.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unify_compounds()
