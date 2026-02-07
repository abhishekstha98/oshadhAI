from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.explain import Explainer

app = FastAPI(
    title="Antigravity API",
    description="Biological Plausibility Scoring for Herbal Formulations",
    version="1.1.0"
)

# Initialize Explainer (Global Load)
# This happens once when the container starts
print("Loading Antigravity Engine...")
explainer = Explainer()
print("Engine Loaded.")

class ScoreRequest(BaseModel):
    smiles_list: List[str]
    weights: Optional[List[str]] = None # "LOW", "MEDIUM", "HIGH" or floats
    ic50_list: Optional[List[float]] = None

@app.get("/")
def health_check():
    return {"status": "active", "version": "1.1.0"}

@app.post("/score")
def score_compounds(req: ScoreRequest):
    """
    Score a list of SMILES strings.
    """
    try:
        # Check for empty input
        if not req.smiles_list:
            raise HTTPException(status_code=400, detail="No SMILES provided")
            
        result = explainer.explain(
            smiles_list=req.smiles_list,
            weights_levels=req.weights, # Logic to handle this needs to be in explainer or here
            ic50_list=req.ic50_list
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
