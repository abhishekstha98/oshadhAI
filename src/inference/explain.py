import sys
from pathlib import Path

# Add project root to path if running as script
if __name__ != "__main__":
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import torch
import logging
import json
import duckdb
from src.model.encoder import MoleculeEncoder
from src.model.reasoner import CombinationReasoner
from src.model.scorer import PlausibilityScorer
from src.model.heads import TargetPredictor
from src.training.ranking import process_batch_smiles
from src.ingest.db import DB_PATH
from src.inference.scoring import normalize_dosage, map_categorical_dosage, apply_dose_weighting, normalize_potency
from src.inference.feedback import FeedbackManager
from src.inference.qsar import QsarPredictor
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors

class Explainer:
    def __init__(self, checkpoints_dir="checkpoints"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Models
        self.encoder = MoleculeEncoder(atom_in_dim=66).to(self.device).eval()
        self.reasoner = CombinationReasoner(embed_dim=256).to(self.device).eval()
        self.scorer = PlausibilityScorer(embed_dim=256).to(self.device).eval()
        
        # Load TargetPredictor for biological grounding
        self.target_predictor = None
        self.num_targets = 0
        
        # Load weights (safe load)
        try:
            pt1 = torch.load(f"{checkpoints_dir}/stage1.pt", map_location=self.device)
            self.encoder.load_state_dict(pt1['encoder'])
            
            # Try to load target predictor
            if 'target_head' in pt1:
                self.num_targets = pt1['target_head']['net.3.weight'].shape[0]
                self.target_predictor = TargetPredictor(embed_dim=256, num_targets=self.num_targets).to(self.device).eval()
                self.target_predictor.load_state_dict(pt1['target_head'])
                logging.info(f"Loaded TargetPredictor with {self.num_targets} targets")
            
            pt3 = torch.load(f"{checkpoints_dir}/stage3.pt", map_location=self.device)
            self.reasoner.load_state_dict(pt3['reasoner'])
            self.scorer.load_state_dict(pt3['scorer'])
            logging.info("Models loaded successfully.")
        except Exception as e:
            logging.warning(f"Could not load checkpoints: {e}. Using random weights for demo.")
        
        # Load calibration data
        self.calibration = self._load_calibration(f"{checkpoints_dir}/calibration.json")
        
        # Load target names
        self.target_names = self._load_target_names()
        
        # Load curated risk allowlists
        self.risk_allowlist = self._load_risk_allowlist()
        
        # Initialize Feedback Manager
        self.feedback_manager = FeedbackManager()
        
        # Load known compounds for novelty check (lazy load)
        self.known_inchi_keys = self._load_known_compounds()

        # Initialize QSAR Predictor
        self.qsar_predictor = QsarPredictor()
        
    def _load_known_compounds(self):
        """Load set of known InChIKeys for novelty detection."""
        try:
            con = duckdb.connect(str(DB_PATH), read_only=True)
            keys = con.execute("SELECT inchi_key FROM compounds").fetchall()
            con.close()
            return set(k[0] for k in keys)
        except Exception as e:
            logging.warning(f"Could not load known compounds: {e}")
            return set()
        
    def _load_risk_allowlist(self):
        """Load curated target allowlists for biological risk signals."""
        path = Path(__file__).parent.parent / "resources" / "risk_target_allowlists.json"
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Could not load risk allowlist: {e}. Using fallback categories.")
            return {
                "cardiovascular": ["KCNH2", "hERG"],
                "metabolic_burden": ["CYP3A4", "CYP2D6"],
                "hepatic_metabolic_burden": ["BSEP", "MRP2"]
            }

    def _load_calibration(self, path):
        """Load percentile calibration data (supports nested format)."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                # Handle new nested format or legacy flat format
                if "percentiles" in data:
                    return data["percentiles"]
                return data
        except FileNotFoundError:
            logging.warning(f"Calibration file not found at {path}. Percentiles unavailable.")
            return None
    
    # Curated Target Allowlists moved to file
    
    # Dose Categorical Mapping (Heuristic Ordinal Scale)
    DOSE_MAP = {
        "LOW": 0.33,
        "MEDIUM": 0.66,
        "HIGH": 1.0
    }
    
    CONFIDENCE_THRESHOLD = 0.20 # τ - Lab-Readiness Enforcement
    
    def _load_target_names(self, db_path=DB_PATH):
        """Load target ID -> name mapping from database, matching training index."""
        try:
            con = duckdb.connect(str(db_path), read_only=True)
            # Reconstruct the exact same list used in training loader
            # loader.py used: SELECT DISTINCT target_id FROM activities
            # For consistency, we should ideally have sorted this, but we must match what was done.
            targets_query = "SELECT DISTINCT target_id FROM activities"
            trained_targets = con.execute(targets_query).fetchall()
            
            # Map target_id to name from the targets metadata library
            names_query = "SELECT target_id, name FROM targets"
            id_to_name = {tid: name for tid, name in con.execute(names_query).fetchall()}
            
            con.close()
            
            # Return {index: name} where index is the position in the training target list
            return {i: id_to_name.get(tid[0], f"ID_{tid[0]}") for i, tid in enumerate(trained_targets)}
        except Exception as e:
            logging.warning(f"Could not load target names: {e}")
            return {}

    def _get_percentile(self, score):
        """Convert score to percentile and decision band."""
        if not self.calibration:
            return None, None, "Calibration data unavailable"
        
        # Find percentile
        percentile = 0
        if score <= self.calibration['p10']:
            percentile = (score - self.calibration['p0']) / (self.calibration['p10'] - self.calibration['p0']) * 10
        elif score <= self.calibration['p25']:
            percentile = 10 + (score - self.calibration['p10']) / (self.calibration['p25'] - self.calibration['p10']) * 15
        elif score <= self.calibration['p50']:
            percentile = 25 + (score - self.calibration['p25']) / (self.calibration['p50'] - self.calibration['p25']) * 25
        elif score <= self.calibration['p75']:
            percentile = 50 + (score - self.calibration['p50']) / (self.calibration['p75'] - self.calibration['p50']) * 25
        elif score <= self.calibration['p90']:
            percentile = 75 + (score - self.calibration['p75']) / (self.calibration['p90'] - self.calibration['p75']) * 15
        else:
            percentile = 90 + (score - self.calibration['p90']) / (self.calibration['p100'] - self.calibration['p90']) * 10
        
        percentile = max(0, min(100, percentile))
        
        # Decision bands
        if percentile < 10:
            band = "DISCARD"
            recommendation = "Strong negative indicators, not recommended for testing"
        elif percentile < 25:
            band = "LOW_PRIORITY"
            recommendation = "Weak plausibility, deprioritize unless novel hypothesis"
        elif percentile < 75:
            band = "INVESTIGATE"
            recommendation = "Moderate plausibility, consider for experimental validation"
        elif percentile < 90:
            band = "HIGH_PRIORITY"
            recommendation = "Strong plausibility, prioritize for testing"
        else:
            band = "CRITICAL"
            recommendation = "Exceptional plausibility, immediate experimental validation recommended"
        
        return percentile, band, recommendation

    def _predict_targets(self, smiles):
        """
        Predict targets for a compound with hard confidence gating (τ = 0.20).
        """
        if not self.target_predictor:
            return []
        
        try:
            emb, mask = process_batch_smiles([[smiles]], self.encoder, self.device)
            if emb is None:
                return []
            
            logits = self.target_predictor(emb[0, 0, :])
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            
            # Gating Logic
            max_prob = float(probs.max())
            top_indices = probs.argsort()[-5:][::-1]
            top_probs = probs[top_indices]
            
            targets = []
            for idx, prob in zip(top_indices, top_probs):
                if prob < self.CONFIDENCE_THRESHOLD:
                    continue  
                
                target_name = self.target_names.get(int(idx), f"ID_{idx}")
                
                # Check confidence level
                if prob >= 0.70:
                    targets.append(target_name)
                else:
                    targets.append(f"{target_name} (low confidence)")
            
            # Record metadata for summary
            return {
                "targets": targets if targets else ["No high-confidence targets"],
                "max_prob": max_prob,
                "count": len(targets)
            }
        except Exception as e:
            logging.warning(f"Target prediction failed: {e}")
            return {"targets": [], "max_prob": 0.0, "count": 0}

    def _calculate_risk_flags(self, smiles_list, target_predictions):
        """Derive risk flags from gated targets against allowlists."""
        risk_flags = {}
        for smiles in smiles_list:
            pred = target_predictions.get(smiles, {"targets": []})
            targets = pred["targets"]
            comp_flags = []
            
            if "No high-confidence targets" in targets:
                risk_flags[smiles] = []
                continue

            for target in targets:
                name = target.split(" (")[0]
                for category, allowlist in self.risk_allowlist.items():
                    if any(a in name for a in allowlist):
                        if category not in comp_flags:
                            comp_flags.append(category)
            
            if pred["count"] >= 10:
                comp_flags.append("high_target_promiscuity")
                
            risk_flags[smiles] = comp_flags
        return risk_flags

    def _calculate_dose_sensitivity(self, smiles_list, weights, base_score):
        """Measure how small changes in weight affect the final score (Finite Difference)."""
        eps = 0.05
        sensitivities = []
        
        for i, smiles in enumerate(smiles_list):
            # Perturb weight
            w_new = list(weights)
            w_new[i] += eps
            
            # Re-normalize
            sum_w = sum(w_new)
            w_norm = [w / sum_w for w in w_new]
            
            # Re-score
            new_res = self.score_set(smiles_list, w_norm)
            new_score = new_res['total_score'] if new_res else base_score
            
            delta = abs(new_score - base_score) / eps
            sensitivities.append({
                "smiles": smiles,
                "sensitivity_score": round(delta, 3),
                "interpretation": "Strongly influences formulation plausibility" if delta > 0.5 else "Moderate influence on formulation ranking"
            })
            
        return sensitivities

    def score_set(self, smiles_list, weights=None, ic50_list=None):
        """
        Returns raw score components and overall score with optional dosage/potency weighting.
        """
        if not smiles_list:
            return None
            
        # 1. Normalize and weight dosage/potency
        weights = normalize_dosage(weights) if weights else [1.0 / len(smiles_list)] * len(smiles_list)
        potencies = normalize_potency(ic50_list) if ic50_list else [1.0 / len(smiles_list)] * len(smiles_list)

        emb, mask = process_batch_smiles([smiles_list], self.encoder, self.device)
        if emb is None: return None
        
        # 2. Apply scaling: h_i' = W_i * h_i
        emb = apply_dose_weighting(emb, weights, potencies, self.device)
        
        formula = self.reasoner(emb, mask=mask)
        components = self.scorer(formula)
        
        # Unpack tensors to floats
        res = {k: v.item() for k, v in components.items()}
        
        # Calculate derived score
        res['total_score'] = res['coverage'] - res['redundancy'] - res['risk']
        return res

    def explain(self, smiles_list, weights=None, weights_levels=None, ic50_list=None):
        """
        Generate full explanation with Lab-Readiness overrides.
        """
        # --- 0. Feedback Check ---
        feedback_entry = self.feedback_manager.get_feedback(smiles_list)
        feedback_boost = 0.0
        feedback_override_band = None
        
        if feedback_entry:
            lbl = feedback_entry["label"]
            if lbl == "VALID": feedback_boost = 0.5
            elif lbl == "SYNERGY": feedback_boost = 1.0
            elif lbl == "INERT": feedback_boost = -0.5
            elif lbl == "TOXIC": 
                feedback_override_band = "DISCARD"
                # Force high risk later

        # 1. Handle Dosage and Potency
        if weights_levels is not None and weights is None:
            weights = map_categorical_dosage(weights_levels)
        
        # Normalize early for consistent contributor analysis
        weights = normalize_dosage(weights) if weights else [1.0 / len(smiles_list)] * len(smiles_list)
        
        # Handle Potency: Fallback to QSAR if IC50 is missing
        potency_sources = {}
        target_potencies = []
        
        if ic50_list:
            target_potencies = normalize_potency(ic50_list)
            for i, s in enumerate(smiles_list):
                potency_sources[s] = "Experimental IC50"
        else:
            # QSAR Prediction Loop
            raw_qsar_scores = []
            for s in smiles_list:
                q_score = self.qsar_predictor.predict_bioactivity(s)
                raw_qsar_scores.append(q_score)
                potency_sources[s] = f"QSAR Estimate (QED={q_score})"
            
            # Normalize QSAR scores to act as relative potency
            total_q = sum(raw_qsar_scores)
            if total_q > 0:
                target_potencies = [q / total_q for q in raw_qsar_scores]
            else:
                target_potencies = [1.0 / len(smiles_list)] * len(smiles_list)

        potencies = target_potencies
        
        # Combined influence vector W_i = normalize(w_i * p_i)
        combined_influences = [w * p for w, p in zip(weights, potencies)]
        combined_influences = normalize_dosage(combined_influences)
            
        # 2. Target Predictions (Gated)
        raw_preds = {s: self._predict_targets(s) for s in smiles_list}
        total_gated_count = sum(p["count"] for p in raw_preds.values())
        
        # 3. Base Score (Weighted)
        base = self.score_set(smiles_list, weights, ic50_list)
        if base is None: return {"error": "Invalid input"}
        
        # Gating Enforcement: Zero coverage if no high-confidence targets
        if total_gated_count == 0:
            base['coverage'] = 0.0
            base['total_score'] = base['coverage'] - base['redundancy'] - base['risk']

        # 4. Decision Policy & Overrides
        percentile, base_band, base_rec = self._get_percentile(base['total_score'])
        
        final_band = base_band
        rule_trace = []
        
        # Apply feedback boost to score BEFORE banding (unless toxic)
        if feedback_entry and feedback_entry["label"] != "TOXIC":
            base['total_score'] += feedback_boost

        if base['uncertainty'] >= 0.85:
            if final_band in ["HIGH_PRIORITY", "CRITICAL"]:
                final_band = "INVESTIGATE"
                rule_trace.append("UNCERTAINTY_OVERRIDE: Extreme sparsity, capping at INVESTIGATE")
        
        # Apply Feedback Override to Band
        if feedback_override_band:
            final_band = feedback_override_band
            rule_trace.append(f"FEEDBACK_OVERRIDE: Lab marked as {feedback_entry['label']}")
        
        if base['risk'] >= 0.80:
            if final_band in ["HIGH_PRIORITY", "CRITICAL"]:
                final_band = "INVESTIGATE"
                rule_trace.append("RISK_OVERRIDE: Elevated biological risk signals, capping at INVESTIGATE")
        
        if base['coverage'] <= 0.02:
            if final_band in ["HIGH_PRIORITY", "CRITICAL", "INVESTIGATE"]:
                final_band = "LOW_PRIORITY"
                rule_trace.append("COVERAGE_OVERRIDE: Near-zero target engagement, mapping to LOW_PRIORITY")
        
        if base['risk'] >= 0.85 and base['uncertainty'] >= 0.85:
            if final_band != "DISCARD":
                final_band = "LOW_PRIORITY"
                rule_trace.append("COMPOUND_SIGNAL_OVERRIDE: High risk + high uncertainty")
            
        if total_gated_count == 0:
            if final_band not in ["DISCARD", "LOW_PRIORITY"]:
                final_band = "LOW_PRIORITY"
                rule_trace.append("TARGET_GATING_OVERRIDE: No high-confidence targets identified")

        # 5. Risk Signals
        compound_risk_flags = self._calculate_risk_flags(smiles_list, raw_preds)
        all_factors = set()
        for flags in compound_risk_flags.values():
            all_factors.update(flags)
            
        factor_map = {
             "cardiovascular": "Predicted cardiovascular risk signals",
             "metabolic_burden": "Inferred metabolic burden (CYP overlap)",
             "hepatic_metabolic_burden": "Hepatic metabolic burden signal",
             "cns": "Predicted central nervous system interaction signal",
             "inflammation": "Inflammatory pathway interaction signal",
             "high_target_promiscuity": "Elevated biological promiscuity"
        }
        primary_risk_factors = [factor_map.get(f, f) for f in all_factors]
        
        # Risk Fallback for Lab-Readiness
        if base['risk'] >= 0.60 and not primary_risk_factors:
            primary_risk_factors.append("Risk driven by high uncertainty / evidence sparsity under known data")

        # FEEDBACK OVERRIDE FOR RISK
        if feedback_entry and feedback_entry["label"] == "TOXIC":
            base['risk'] = 1.0
            primary_risk_factors.insert(0, "Lab Validation Failure: Marked TOXIC by tester.")

        # 6. Contributor Analysis (Ablation with Weights)
        sum_w = sum(weights)
        norm_weights = {s: round(w/sum_w, 2) for s, w in zip(smiles_list, weights)}
        
        contributor_details = []
        for i, smile in enumerate(smiles_list):
            subset = smiles_list[:i] + smiles_list[i+1:]
            sub_w = weights[:i] + weights[i+1:]
            
            if not subset:
                sub_score = 0
            else:
                sub_res = self.score_set(subset, sub_w)
                # Gating check for subset (optional but consistent)
                sub_score = sub_res['total_score'] if sub_res else 0
            
            contribution = base['total_score'] - sub_score
            comp_pred = raw_preds.get(smile, {"targets": []})
            
            contributor_details.append({
                "smiles": smile,
                "contribution": round(contribution, 3),
                "weight": norm_weights[smile],
                "top_targets": comp_pred["targets"][:3] if comp_pred["targets"] else ["Found in baseline"],
                "risk_flags": compound_risk_flags.get(smile, [])
            })
        
        contributor_details.sort(key=lambda x: x['contribution'], reverse=True)

        # 7. Final Output Assembly
        output = {
            "plausibility_score": round(base['total_score'], 3),
            "metrics": {
                "coverage": round(base['coverage'], 3),
                "redundancy_penalty": round(base['redundancy'], 3),
                "risk_penalty": round(base['risk'], 3),
                "uncertainty": round(base['uncertainty'], 3)
            },
            "uncertainty_explained": {
                "total": round(base['uncertainty'], 3),
                "sources": {
                    "data_sparsity": round(base.get('u_sparsity', 0.0), 3),
                    "prediction_variance": round(base.get('u_variance', 0.0), 3),
                    "out_of_distribution": round(base.get('u_ood', 0.0), 3)
                },
                "interpretation": self._explain_uncertainty(base['uncertainty'])
            },
            "target_confidence_summary": {
                "threshold": self.CONFIDENCE_THRESHOLD,
                "max_prob_per_compound": {s: round(p["max_prob"], 3) for s, p in raw_preds.items()},
                "gated_target_counts": {s: p["count"] for s, p in raw_preds.items()},
                "note": "Targets below threshold are suppressed to prevent overinterpretation in lab settings."
            },
            "decision": {
                "band": final_band,
                "percentile": round(percentile, 1) if percentile else 0.0,
                "recommendation": base_rec if final_band == base_band else "Review formulation: Plausibility capped by risk or uncertainty constraints.",
                "rule_trace": rule_trace if rule_trace else ["Applied: PERCENTILE_MAPPING"]
            },
            "dose_analysis": {
                "input_mode": "normalized_weight_and_potency",
                "normalized_weights": {s: round(w, 2) for s, w in zip(smiles_list, weights)},
                "normalized_potencies": {s: round(p, 2) for s, p in zip(smiles_list, potencies)},
                "activity_source": potency_sources,
                "combined_influence": {s: round(ci, 2) for s, ci in zip(smiles_list, combined_influences)},
                "ic50_adjustment_explanation": "Combined influence scales compound embeddings by both relative dosage and biological potency. QSAR estimates (QED) are used when experimental IC50 is unavailable."
            },
            "biological_risk_assessment": {
                "risk_penalty": round(base['risk'], 3),
                "risk_level": "HIGH" if base['risk'] > 0.7 else "MODERATE" if base['risk'] > 0.4 else "LOW",
                "primary_risk_factors": primary_risk_factors,
                "compound_level_flags": compound_risk_flags,
                "interpretation": "Biological risk signals suggest selective metabolic or off-target concerns."
            },
            "contributors": contributor_details,
            "biological_context": {
                "predicted_targets": list(set([t for p in raw_preds.values() for t in p["targets"] if t != "No high-confidence targets"]))[:5],
                "coverage_interpretation": "Broad biological engagement identified" if total_gated_count >= 3 else "No high-confidence target engagement identified under known data."
            },
            "adme_simulation": self._compute_adme_placeholders(smiles_list),
            "novelty_report": self._compute_novelty(smiles_list),
            "feedback_status": {
                "has_feedback": bool(feedback_entry),
                "label": feedback_entry["label"] if feedback_entry else None,
                "score_adjustment": feedback_boost if feedback_entry else 0.0
            },
            "disclaimer": "This score represents plausibility ranking under known compound-target biology. It does NOT predict biological reinforcement, therapeutic results, or human health outcomes. Experimental validation is required."
        }
        
        return output

    def _compute_adme_placeholders(self, smiles_list):
        """Compute ADME proxies (LogP) and placeholders."""
        adme_data = []
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            logP = Crippen.MolLogP(mol) if mol else 0.0
            mw = Descriptors.MolWt(mol) if mol else 0.0
            # Lipinski-like proxies
            solubility_proxy = "High" if logP < 3 else "Moderate" if logP < 5 else "Low"
            bbb_proxy = "Permeable" if (logP > 0 and logP < 3 and mw < 450) else "Impermeable" # simplified
            
            adme_data.append({
                "smiles": s,
                "logP_proxy": round(logP, 2),
                "solubility_class": f"{solubility_proxy} (Simulated)",
                "blood_brain_barrier": f"{bbb_proxy} (Simulated)",
                "metabolic_stability": "Unknown (Placeholder)"
            })
        return adme_data

    def _compute_novelty(self, smiles_list):
        """Check if compounds are known in the database."""
        known_count = 0
        novel_compounds = []
        
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            if not mol: continue
            ikey = Chem.MolToInchiKey(mol)
            if ikey in self.known_inchi_keys:
                known_count += 1
            else:
                novel_compounds.append(s)
                
        return {
            "known_compounds": known_count,
            "novel_compounds": len(novel_compounds),
            "novel_smiles": novel_compounds,
            "inference_method": "Graph Isomorphism Network (Generalization)" if novel_compounds else "Database Lookup"
        }

    def _percentile_desc(self, p):
        """Describe percentile in human terms."""
        if p < 10:
            return "bottom 10%"
        elif p < 25:
            return "bottom quartile"
        elif p < 50:
            return "lower half"
        elif p < 75:
            return "upper half"
        elif p < 90:
            return "top quartile"
        else:
            return "top 10%"

    def _explain_uncertainty(self, uncertainty):
        """Generate textual uncertainty explanation."""
        if uncertainty > 1.5:
            return "Very high uncertainty - predictions should be treated with extreme caution due to limited ChEMBL evidence"
        elif uncertainty > 1.0:
            return "High uncertainty - limited evidence density in training data for these compounds"
        elif uncertainty > 0.5:
            return "Moderate uncertainty - some predictions based on sparse data"
        else:
            return "Low uncertainty - compounds well-represented in ChEMBL training data"

    def _generate_suggestions(self, contributions, metrics, bio_context):
        """Generate conservative counterfactual suggestions."""
        suggestions = []
        
        # Negative contributors
        for smile, contrib in contributions.items():
            if contrib < -0.5:
                suggestions.append(
                    f"Consider removing compound with contribution {contrib:.2f} (may reduce risk or redundancy)"
                )
        
        # High redundancy
        if metrics['redundancy'] > 0.7:
            suggestions.append(
                f"High target overlap detected ({bio_context['target_overlap_count']} overlapping targets) - "
                "consider alternatives with distinct mechanisms"
            )
        
        # Low coverage
        if metrics['coverage'] < 0.3:
            suggestions.append(
                "Low target diversity - consider adding compounds with novel biological targets"
            )
        
        # High risk
        if metrics['risk'] > 0.7:
            suggestions.append(
                "Significant risk flags present - review toxicity profiles before experimental testing"
            )
        
        return suggestions if suggestions else ["No specific improvement suggestions - formulation appears balanced"]

    def _interpret(self, metrics):
        """
        Textual interpretation of plausibility metrics.
        CRITICAL: This is NOT an efficacy or synergy claim. 
        Output describes biological coherence under known experimental data.
        """
        msgs = []
        if metrics['coverage'] > 0.7: 
            msgs.append("High biological target coverage detected.")
        elif metrics['coverage'] < 0.3: 
            msgs.append("Low target coverage - limited biological diversity under known data.")
        
        if metrics['redundancy'] > 0.5: 
            msgs.append("High redundancy detected - overlapping target predictions.")
        
        if metrics['risk'] > 0.5: 
            msgs.append("Significant toxicity or metabolic risk signals present.")
        
        if metrics['uncertainty'] > 1.0: 
            msgs.append("Prediction highly uncertain due to limited evidence density.")
        
        return " ".join(msgs) if msgs else "Moderate plausibility profile."
