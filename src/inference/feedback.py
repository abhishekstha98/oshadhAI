import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional

FEEDBACK_FILE = Path("data/feedback.json")

class FeedbackManager:
    """
    Manages lab tester feedback for formulation validation.
    Stores feedback keyed by the canonical hash of the compound set.
    """
    def __init__(self, storage_path: Path = FEEDBACK_FILE):
        self.storage_path = storage_path
        self.ensure_storage()

    def ensure_storage(self):
        """Ensure feedback file exists."""
        if not self.storage_path.parent.exists():
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.storage_path.exists():
            with open(self.storage_path, 'w') as f:
                json.dump({}, f)

    def _get_hash(self, smiles_list: List[str]) -> str:
        """Generate a consistent hash for a set of compounds."""
        # Sort to ensure set-invariance
        sorted_smiles = sorted([s.strip() for s in smiles_list])
        concat_str = "|".join(sorted_smiles)
        return hashlib.sha256(concat_str.encode()).hexdigest()

    def submit_feedback(self, smiles_list: List[str], label: str, notes: str = ""):
        """
        Submit feedback for a formulation.
        Labels: VALID, TOXIC, SYNERGY, INERT
        """
        if label not in ["VALID", "TOXIC", "SYNERGY", "INERT"]:
            raise ValueError(f"Invalid label: {label}. Must be VALID, TOXIC, SYNERGY, or INERT.")

        key = self._get_hash(smiles_list)
        
        entry = {
            "smiles": smiles_list,
            "label": label,
            "notes": notes,
            "timestamp": datetime.now().isoformat()
        }

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}

        data[key] = entry

        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logging.info(f"Feedback stored for formulation {key[:8]}... as {label}")

    def get_feedback(self, smiles_list: List[str]) -> Optional[dict]:
        """Retrieve feedback for a formulation if it exists."""
        key = self._get_hash(smiles_list)
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            return data.get(key)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
