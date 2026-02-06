import duckdb
from src.ingest.db import DB_PATH
import logging

def generate_audit(db_path=DB_PATH):
    con = duckdb.connect(str(db_path), read_only=True)
    
    report = []
    report.append("=== ANTIGRAVITY DATA MANIFEST ===")
    
    # Counts
    n_compounds = con.execute("SELECT count(*) FROM compounds").fetchone()[0]
    n_targets = con.execute("SELECT count(*) FROM targets").fetchone()[0]
    n_activities = con.execute("SELECT count(*) FROM activities").fetchone()[0]
    n_herbs = con.execute("SELECT count(*) FROM herbs").fetchone()[0]
    
    report.append(f"Total Compounds (Raw): {n_compounds}")
    report.append(f"Total Targets (ChEMBL): {n_targets}")
    report.append(f"Total Activities: {n_activities}")
    report.append(f"Total Herbs: {n_herbs}")
    
    # Breakdown by Source
    report.append("\nCompound Source Breakdown:")
    rows = con.execute("SELECT source, count(*) FROM compounds GROUP BY source").fetchall()
    for source, cnt in rows:
         report.append(f"  - {source}: {cnt}")

    # Unification Stats
    try:
        n_master = con.execute("SELECT count(*) FROM master_compounds").fetchone()[0]
        report.append(f"\nUnified Master Compounds: {n_master}")
        
        # Overlap
        multi_source = con.execute("SELECT count(*) FROM master_compounds WHERE len(sources_list) > 1").fetchone()[0]
        report.append(f"Compounds with Multi-Source Support: {multi_source}")
    except:
        report.append("\n(Master compounds not unified yet)")

    # Risk Flags
    try:
        n_risk = con.execute("SELECT count(*) FROM risk_flags").fetchone()[0]
        report.append(f"\nTotal Risk Flags (Tox21): {n_risk}")
    except:
        pass

    manifest_path = "manifest.txt"
    with open(manifest_path, "w") as f:
        f.write("\n".join(report))
        
    logging.info(f"Manifest written to {manifest_path}")
    print("\n".join(report))

if __name__ == "__main__":
    generate_audit()
