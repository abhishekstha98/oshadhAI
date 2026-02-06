from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal

class Compound(BaseModel):
    """
    Contract for compound data.
    Source of truth for chemical entities.
    """
    compound_id: str = Field(..., description="Unique identifier (e.g., ChEMBL ID or internal hash)")
    smiles: str = Field(..., description="Canonical SMILES string")
    inchi_key: str = Field(..., description="Standard InChI Key")
    source: str = Field(..., description="Source database (e.g., 'chembl', 'tcm')")

    @field_validator('smiles')
    def validate_smiles(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("SMILES cannot be empty")
        return v

class Target(BaseModel):
    """
    Contract for biological targets.
    """
    target_id: str = Field(..., description="Unique identifier (e.g., ChEMBL Target ID)")
    name: str = Field(..., description="Human readable name of the target")
    organism: str = Field(..., description="Organism (e.g., 'Homo sapiens')")

class Activity(BaseModel):
    """
    Contract for compound-target interactions.
    """
    compound_id: str = Field(..., description="Foreign key to Compound")
    target_id: str = Field(..., description="Foreign key to Target")
    p_activity: Optional[float] = Field(None, description="Negative log activity (pIC50, pKi, etc.), if available")
    assay_confidence: int = Field(..., ge=0, le=9, description="ChEMBL confidence score (9=highest)")
    quality_weight: float = Field(1.0, ge=0.0, le=1.0, description="Weight derived from assay parameters")

class Herb(BaseModel):
    """
    Contract for herbal entities.
    """
    herb_id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Botanical or common name")
    source: str = Field(..., description="Source dataset")

class HerbCompound(BaseModel):
    """
    Mapping between herbs and compounds.
    """
    herb_id: str = Field(..., description="Foreign key to Herb")
    compound_id: str = Field(..., description="Foreign key to Compound")
    evidence_weight: float = Field(0.5, ge=0.0, le=1.0, description="Confidence of presence detection")
