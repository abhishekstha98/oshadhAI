import pandas as pd
import random
from pathlib import Path

def generate_dummy():
    root = Path("datasets/ChEMBL")
    p_comp = root / "chembl_compounds.csv"
    p_targ = root / "chembl_targets.csv"
    p_act = root / "chembl_activity.csv"
    
    if not (p_comp.exists() and p_targ.exists()):
        print("Missing compounds or targets to link.")
        return

    # Load IDs
    comps = pd.read_csv(p_comp, usecols=['compound_id'])['compound_id'].tolist()
    targs = pd.read_csv(p_targ, usecols=['target_id'])['target_id'].tolist()
    
    print(f"Found {len(comps)} compounds and {len(targs)} targets.")
    
    # Generate 1000 dummy activities
    data = []
    for _ in range(5000):
        c = random.choice(comps)
        t = random.choice(targs)
        p = round(random.uniform(4.0, 10.0), 2)
        data.append({'compound_id': c, 'target_id': t, 'pActivity': p})
        
    df = pd.DataFrame(data)
    df.to_csv(p_act, index=False)
    print(f"Generated {len(df)} dummy activities to {p_act}")

if __name__ == "__main__":
    generate_dummy()
