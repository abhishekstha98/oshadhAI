from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
import sys
from pathlib import Path
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.explain import Explainer

app = FastAPI(
    title="Antigravity API",
    description="Biological Plausibility Scoring for Herbal Formulations",
    version="1.2.0"
)

# Initialize Explainer (Global Load)
logger.info("Initializing Antigravity Engine...")
try:
    explainer = Explainer()
    logger.info("Engine Loaded Successfully.")
except Exception as e:
    logger.error(f"Failed to load engine: {e}")
    # We don't crash here so that the container can start and report health
    explainer = None

class ScoreRequest(BaseModel):
    smiles_list: List[str]
    weights: Optional[List[float]] = None
    weights_levels: Optional[List[str]] = None # "LOW", "MEDIUM", "HIGH"
    ic50_list: Optional[List[float]] = None

@app.get("/")
def health_check():
    status = "active" if explainer else "error"
    return {"status": status, "version": "1.2.0"}

@app.post("/score")
def score_compounds(req: ScoreRequest):
    """
    Score a list of SMILES strings.
    """
    if not explainer:
        raise HTTPException(status_code=503, detail="Inference engine not initialized.")

    try:
        # Check for empty input
        if not req.smiles_list:
            raise HTTPException(status_code=400, detail="No SMILES provided")
            
        result = explainer.explain(
            smiles_list=req.smiles_list,
            weights=req.weights,
            weights_levels=req.weights_levels,
            ic50_list=req.ic50_list
        )
        
        return result
    except Exception as e:
        logger.error(f"Scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
