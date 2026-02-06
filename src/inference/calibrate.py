"""
Calibration script for Antigravity plausibility scoring.

Generates reference distribution from stratified sampling of herb combinations.
Computes percentile bins for ranking context.

USER CONDITIONS:
- Stratified sampling by combination size
- Exclude trivial combinations (single-compound)
- Diverse herb representation
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import duckdb
import random
import json
import logging
from src.ingest.db import DB_PATH
from src.inference.explain import Explainer

logging.basicConfig(level=logging.INFO)

def sample_stratified_combinations(db_path=DB_PATH, total_samples=1000):
    """
    Sample herb combinations stratified by size.
    
    Stratification:
    - 2-compound: 30%
    - 3-compound: 30%
    - 4-compound: 20%
    - 5+ compound: 20%
    
    Excludes:
    - Single-compound (trivial)
    - Empty sets
    
    Ensures diverse herb representation.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    
    # Get all herbs and their compounds
    herbs_data = {}
    rows = con.execute("""
        SELECT hc.herb_id, coalesce(m.canonical_smiles, c.smiles) as smiles
        FROM herb_compounds hc
        JOIN compounds c ON hc.compound_id = c.compound_id
        LEFT JOIN master_compounds m ON c.inchi_key = m.inchi_key
        WHERE c.smiles IS NOT NULL OR m.canonical_smiles IS NOT NULL
    """).fetchall()
    
    for herb_id, smiles in rows:
        if herb_id not in herbs_data:
            herbs_data[herb_id] = []
        herbs_data[herb_id].append(smiles)
    
    herb_ids = list(herbs_data.keys())
    logging.info(f"Found {len(herb_ids)} herbs with compound mappings")
    
    # Stratified sampling
    strata = {
        2: int(total_samples * 0.30),
        3: int(total_samples * 0.30),
        4: int(total_samples * 0.20),
        5: total_samples - int(total_samples * 0.80)  # Remainder for 5+
    }
    
    combinations = []
    
    for size, count in strata.items():
        logging.info(f"Sampling {count} combinations of size {size}")
        
        for _ in range(count):
            # Sample random herbs (ensure diversity - no repeats in single combination)
            if size <= len(herb_ids):
                sampled_herbs = random.sample(herb_ids, size)
            else:
                sampled_herbs = random.choices(herb_ids, k=size)
            
            # Get 1 random compound from each herb
            smiles_list = []
            for hid in sampled_herbs:
                if herbs_data[hid]:
                    smiles_list.append(random.choice(herbs_data[hid]))
            
            if len(smiles_list) >= 2:  # Exclude trivial
                combinations.append(smiles_list)
    
    con.close()
    return combinations

def score_combinations(combinations, explainer):
    """Score all combinations and return distribution."""
    scores = []
    
    logging.info(f"Scoring {len(combinations)} combinations...")
    
    for i, combo in enumerate(combinations):
        try:
            result = explainer.score_set(combo)
            if result:
                scores.append(result['total_score'])
            
            if (i + 1) % 100 == 0:
                logging.info(f"Scored {i+1}/{len(combinations)}")
        except Exception as e:
            logging.warning(f"Failed to score combination {i}: {e}")
    
    return scores

def compute_percentiles(scores):
    """Compute percentile bins."""
    import numpy as np
    
    scores = sorted(scores)
    
    percentiles = {
        'p0': float(np.percentile(scores, 0)),
        'p10': float(np.percentile(scores, 10)),
        'p25': float(np.percentile(scores, 25)),
        'p50': float(np.percentile(scores, 50)),
        'p75': float(np.percentile(scores, 75)),
        'p90': float(np.percentile(scores, 90)),
        'p100': float(np.percentile(scores, 100)),
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'count': len(scores)
    }
    
    return percentiles

def save_calibration(percentiles, sampling_info, output_path="checkpoints/calibration.json"):
    """Save calibration data with sampling metadata."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "percentiles": percentiles,
        "sampling_strategy": sampling_info,
        "note": "Reference distribution generated via stratified sampling (2-5 compounds)."
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logging.info(f"Calibration saved to {output_path}")

def main():
    """Run calibration workflow."""
    logging.info("=== Antigravity Calibration ===")
    
    # Step 1: Stratified sampling
    logging.info("Step 1: Stratified sampling of herb combinations")
    combinations = sample_stratified_combinations(total_samples=1000)
    logging.info(f"Sampled {len(combinations)} valid combinations")
    
    # Step 2: Load models
    logging.info("Step 2: Loading models")
    explainer = Explainer()
    
    # Step 3: Score all combinations
    logging.info("Step 3: Scoring combinations")
    scores = score_combinations(combinations, explainer)
    logging.info(f"Successfully scored {len(scores)} combinations")
    
    # Step 4: Compute percentiles
    logging.info("Step 4: Computing percentile distribution")
    percentiles = compute_percentiles(scores)
    
    # Step 5: Save
    logging.info("Step 5: Saving calibration data")
    sampling_info = {
        "total_samples": len(scores),
        "strata_targets": {2: 0.3, 3: 0.3, 4: 0.2, 5: 0.2}
    }
    save_calibration(percentiles, sampling_info)
    
    # Summary
    logging.info("\n=== Calibration Complete ===")
    logging.info(f"Score range: [{percentiles['p0']:.3f}, {percentiles['p100']:.3f}]")
    logging.info(f"Median: {percentiles['p50']:.3f}")
    logging.info(f"Samples: {percentiles['count']}")

if __name__ == "__main__":
    main()
