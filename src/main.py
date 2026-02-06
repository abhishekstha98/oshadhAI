import argparse
import logging
import sys
from pathlib import Path

# Adjust path to find src
sys.path.append(str(Path(__file__).parent.parent))

def main():
    parser = argparse.ArgumentParser(description="Antigravity: Pre-experimental Herb Plausibility Screener")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest data from datasets directory")
    ingest_parser.add_argument("--chembl", help="Directory containing ChEMBL CSVs")
    ingest_parser.add_argument("--herbs", help="Directory containing HERB CSVs")
    ingest_parser.add_argument("--tcm", help="Directory containing TCM Excel files")
    ingest_parser.add_argument("--tox21", help="Directory containing Tox21 SDF")
    ingest_parser.add_argument("--mysql", action='store_true', help="Ingest from local MySQL (Python Buffered)")
    ingest_parser.add_argument("--mysql-direct", action='store_true', help="Ingest from local MySQL (Direct DuckDB Attach - FAST)")
    
    # Unify
    unify_parser = subparsers.add_parser("unify", help="Run Compound Unification")
    
    # Train
    train_parser = subparsers.add_parser("train-stage1", help="Train Compound Encoder")
    train_parser.add_argument("--epochs", type=int, default=10)
    
    train3_parser = subparsers.add_parser("train-stage3", help="Train Ranking Model")
    train3_parser.add_argument("--epochs", type=int, default=10)
    
    # Score
    score_parser = subparsers.add_parser("score", help="Score a combination")
    score_parser.add_argument("smiles", nargs="+", help="List of SMILES strings")
    score_parser.add_argument("--weights", type=float, nargs="+", help="Relative weights for each compound (must match SMILES count)")
    score_parser.add_argument("--weights-levels", nargs="+", help="Categorical weights: LOW, MEDIUM, HIGH (must match SMILES count)")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.command == "ingest":
        from src.ingest.db import init_db
        init_db()
        
        if args.chembl:
            from src.ingest.chembl import ingest_chembl_csv
            ingest_chembl_csv(args.chembl)
            
        if args.mysql:
            from src.ingest.chembl_sql import ingest_chembl_sql
            ingest_chembl_sql()
            
        if args.mysql_direct:
            from src.ingest.chembl_direct import ingest_chembl_direct
            ingest_chembl_direct()
            
        if args.herbs:
            from src.ingest.herb import ingest_herb_csv
            ingest_herb_csv(args.herbs)
            
        if args.tcm:
            from src.ingest.tcm import ingest_tcm_excel
            ingest_tcm_excel(args.tcm)
            
        if args.tox21:
            from src.ingest.tox21 import ingest_tox21_sdf
            ingest_tox21_sdf(args.tox21)
            
    elif args.command == "unify":
        from src.ingest.unify import unify_compounds
        unify_compounds()
            
    elif args.command == "train-stage1":
        from src.training.pretrain import train_stage1
        train_stage1(epochs=args.epochs)
        
    elif args.command == "train-stage3":
        from src.training.ranking import train_stage3
        train_stage3(epochs=args.epochs)
        
    elif args.command == "score":
        from src.inference.explain import Explainer
        import json
        
        explainer = Explainer()
        result = explainer.explain(args.smiles, weights=args.weights, weights_levels=args.weights_levels)
        
        print(json.dumps(result, indent=2))
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
