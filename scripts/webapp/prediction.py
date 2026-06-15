import pandas as pd
import numpy as np
import streamlit as st
import json
import re
import subprocess
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from config import DEFAULT_MODEL, VALIDATION_CONFIG_PATH
import logging
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from api import get_chembl_info, get_formula, get_smiles
from utils import calculate_padel_descriptors, calculate_padel_descriptors_batch

logger = logging.getLogger(__name__)
FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_EMBEDDING_MODEL_CACHE = None

# PADEL_JAR_PATH = '../../notebooks/padel.sh'  # Adjust if needed

def sanitize_descriptors(descr_df):
    """Remove or replace NaN and Inf values in descriptor DataFrame"""
    try:
        # Replace infinity values with NaN
        descr_df = descr_df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with column median or 0 if all NaN
        for col in descr_df.columns:
            if descr_df[col].isna().all():
                descr_df[col] = 0.0
            else:
                descr_df[col].fillna(descr_df[col].median(), inplace=True)
        
        return descr_df
    except Exception as e:
        logger.error(f"Error sanitizing descriptors: {str(e)}")
        return descr_df

def _repo_root():
    return Path(__file__).resolve().parents[2]

def _validation_config_path():
    return (Path(__file__).resolve().parent / VALIDATION_CONFIG_PATH).resolve()

def _load_validation_config():
    import sys
    repo = _repo_root()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from brainroute_ml_validation.src.utils import load_config
    return load_config(_validation_config_path())

def _canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

def _prepare_padel_features_from_df(padel_df):
    padel_df = padel_df.drop(columns=['Name'], errors='ignore')
    padel_df = padel_df.apply(pd.to_numeric, errors='coerce')
    padel_df = padel_df.replace([np.inf, -np.inf], np.nan)
    return padel_df.add_prefix("padel__")

def _calculate_morgan_features(smiles_list, cfg):
    import sys
    repo = _repo_root()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from brainroute_ml_validation.src.chemistry import calculate_morgan_matrix

    canonical = [_canonical_smiles(smiles) for smiles in smiles_list]
    morgan = cfg.get("morgan", {})
    n_bits = int(morgan.get("nBits", 2048))
    matrix, valid_positions, _ = calculate_morgan_matrix(
        canonical,
        radius=int(morgan.get("radius", 2)),
        n_bits=n_bits,
        use_chirality=bool(morgan.get("useChirality", True)),
    )
    out = pd.DataFrame(0, index=range(len(smiles_list)), columns=[f"morgan__morgan_{i}" for i in range(n_bits)])
    if len(valid_positions):
        out.iloc[valid_positions, :] = matrix
    return out

def _load_embedding_model(cfg):
    global _EMBEDDING_MODEL_CACHE
    if _EMBEDDING_MODEL_CACHE is not None:
        return _EMBEDDING_MODEL_CACHE

    import os
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_TORCH", "1")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    import torch
    from transformers import AutoModel, AutoTokenizer

    emb_cfg = cfg.get("pretrained_embeddings", {})
    model_name = emb_cfg.get("model_name", "DeepChem/ChemBERTa-77M-MLM")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    _EMBEDDING_MODEL_CACHE = (tokenizer, model, device)
    return _EMBEDDING_MODEL_CACHE

def _calculate_embedding_features(smiles_list, cfg):
    import torch

    tokenizer, model, device = _load_embedding_model(cfg)
    emb_cfg = cfg.get("pretrained_embeddings", {})
    batch_size = int(emb_cfg.get("batch_size", 32))
    max_length = int(emb_cfg.get("max_length", 256))
    pooling = emb_cfg.get("pooling", "cls")
    canonical = [_canonical_smiles(smiles) or str(smiles) for smiles in smiles_list]
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(canonical), batch_size):
            batch = canonical[start : start + batch_size]
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            tokens = {key: value.to(device) for key, value in tokens.items()}
            outputs = model(**tokens)
            hidden = outputs.last_hidden_state
            if pooling == "mean":
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = hidden[:, 0, :]
            embeddings.append(pooled.cpu().numpy())
    arr = np.vstack(embeddings)
    return pd.DataFrame(arr, columns=[f"emb__emb_{i}" for i in range(arr.shape[1])])

def _build_webapp_feature_views(smiles_list):
    cfg = _load_validation_config()
    padel_df, failed_indices = calculate_padel_descriptors_batch(smiles_list)
    padel_features = _prepare_padel_features_from_df(padel_df)
    morgan_features = _calculate_morgan_features(smiles_list, cfg)
    views = {
        "padel_morgan": pd.concat(
            [padel_features.reset_index(drop=True), morgan_features.reset_index(drop=True)],
            axis=1,
        )
    }
    embedding_error = None
    try:
        embedding_features = _calculate_embedding_features(smiles_list, cfg)
        views["padel_morgan_embeddings"] = pd.concat(
            [
                padel_features.reset_index(drop=True),
                morgan_features.reset_index(drop=True),
                embedding_features.reset_index(drop=True),
            ],
            axis=1,
        )
    except Exception as exc:
        embedding_error = str(exc)
        logger.warning("Embedding feature calculation failed; embedding model will be skipped: %s", exc)
    return views, set(failed_indices), embedding_error

def _model_expected_features(model):
    finite = getattr(model, "named_steps", {}).get("finite") if hasattr(model, "named_steps") else None
    if finite is not None and hasattr(finite, "feature_names_in_"):
        return list(finite.feature_names_in_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return None

def _predict_from_feature_views(feature_views, row_index, models):
    model_feature_views = models.get("model_feature_views", {})
    predictions = {}
    confidences = {}
    positive_probabilities = {}
    errors = {}

    for model_name, model in models.items():
        if model_name in {"feature_names", "model_feature_views"}:
            continue
        feature_view = model_feature_views.get(model_name, "padel_morgan")
        X_view = feature_views.get(feature_view)
        if X_view is None:
            errors[model_name] = f"Feature view unavailable: {feature_view}"
            continue
        expected_features = _model_expected_features(model)
        X_single = X_view.iloc[[row_index]].copy()
        if expected_features:
            X_single = X_single.reindex(columns=expected_features, fill_value=np.nan)
        try:
            pred = model.predict(X_single)
            pred_value = int(pred[0])
            predictions[model_name] = pred_value
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_single)
                if proba.shape[1] > 1:
                    p_pos = float(proba[0, 1])
                    confidence = float(proba[0, pred_value]) * 100
                else:
                    p_pos = float(proba[0, 0])
                    confidence = max(p_pos, 1 - p_pos) * 100
            else:
                p_pos = float(pred_value)
                confidence = None
            positive_probabilities[model_name] = p_pos
            confidences[model_name] = confidence
        except Exception as exc:
            errors[model_name] = str(exc)
            logger.warning("Model %s prediction failed: %s", model_name, exc)

    if not predictions:
        return None, errors

    avg_positive_probability = float(np.mean(list(positive_probabilities.values())))
    ensemble_class = 1 if avg_positive_probability >= 0.5 else 0
    ensemble_pred = "BBB+" if ensemble_class == 1 else "BBB-"
    avg_confidence = max(avg_positive_probability, 1 - avg_positive_probability) * 100
    pred_values = list(predictions.values())
    agreement = sum(1 for pred in pred_values if pred == ensemble_class) / len(pred_values) * 100
    uncertainty = float(np.std(list(positive_probabilities.values())) * 100) if len(positive_probabilities) > 1 else abs(50 - avg_confidence)

    return {
        "predictions": predictions,
        "confidences": confidences,
        "positive_probabilities": positive_probabilities,
        "ensemble_pred": ensemble_pred,
        "avg_confidence": avg_confidence,
        "agreement": agreement,
        "uncertainty": uncertainty,
        "num_models": len(predictions),
        "model_errors": errors,
    }, None

def predict_bbb_penetration_with_uncertainty(mol, models):
    """Predict BBB penetration with uncertainty quantification using multiple models"""
    try:
        # Calculate descriptors
        descr = pd.DataFrame([Descriptors.CalcMolDescriptors(mol)])
        
        # Sanitize descriptors to handle NaN/Inf values
        descr = sanitize_descriptors(descr)
        
        # Scale descriptors
        if 'scaler' not in models:
            raise ValueError("Scaler not found in loaded models")
        
        scaler = models['scaler']
        descr_scaled = scaler.transform(descr)
        
        # Check for NaN/Inf in scaled descriptors
        if np.any(np.isnan(descr_scaled)) or np.any(np.isinf(descr_scaled)):
            logger.warning("Scaled descriptors contain NaN or Inf values, sanitizing...")
            descr_scaled = np.nan_to_num(descr_scaled, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Get predictions from all available models
        predictions = {}
        confidences = {}
        
        for model_name in ['KNN', 'XGB', 'RF', 'LR', 'ET']:  # Add more models as available
            if model_name in models:
                try:
                    model = models[model_name]
                    pred = model.predict(descr_scaled)
                    confidence = model.predict_proba(descr_scaled)
                    
                    predictions[model_name] = pred[0]
                    confidences[model_name] = confidence[0]
                except Exception as e:
                    logger.warning(f"Model {model_name} failed: {str(e)}")
                    continue
        
        if not predictions:
            raise ValueError("All models failed to make prediction")
        
        # Calculate ensemble prediction and uncertainty
        pred_probs = np.array([confidences[model] for model in predictions.keys()])
        
        # Average probabilities across models
        avg_probs = np.mean(pred_probs, axis=0)
        ensemble_pred = "BBB+" if avg_probs[1] > 0.5 else "BBB-"
        ensemble_confidence = max(avg_probs) * 100
        
        # Calculate uncertainty metrics
        std_probs = np.std(pred_probs, axis=0)
        uncertainty = np.max(std_probs) * 100  # Uncertainty as std dev of probabilities
        
        # Agreement between models (what % agree)
        pred_classes = [predictions[model] for model in predictions.keys()]
        agreement = (np.sum(pred_classes) / len(pred_classes)) * 100 if pred_classes else 0
        if agreement > 50:
            agreement = agreement
        else:
            agreement = 100 - agreement
        
        return {
            'prediction': ensemble_pred,
            'confidence': ensemble_confidence,
            'uncertainty': uncertainty,
            'agreement': agreement,
            'individual_predictions': predictions,
            'individual_confidences': confidences,
            'num_models': len(predictions)
        }, None
        
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        logger.error(error_msg)
        return None, error_msg

def predict_bbb_penetration(mol, models):
    """Legacy function for backward compatibility"""
    result, error = predict_bbb_penetration_with_uncertainty(mol, models)
    if error:
        return None, None, None, error
    
    # Return primary model result for compatibility
    primary_model = 'XGB' if 'XGB' in result['individual_predictions'] else list(result['individual_predictions'].keys())[0]
    primary_confidence = result['individual_confidences'][primary_model][result['individual_predictions'][primary_model]] * 100
    
    return result['prediction'], primary_confidence, primary_model, None

def _is_missing(value):
    return value is None or pd.isna(value)

def _get_polarity_bin(tpsa):
    if _is_missing(tpsa):
        return None
    if tpsa < 30:
        return "apolar"
    if tpsa < 60:
        return "nonpolar"
    if tpsa < 100:
        return "moderate"
    return "polar"

def _get_lipophilicity_bin(logp):
    if _is_missing(logp):
        return None
    if logp < -1:
        return "hydrophilic"
    if logp < 1:
        return "balanced"
    if logp < 3:
        return "moderate"
    return "lipophilic"

def _get_size_bin(mw):
    if _is_missing(mw):
        return None
    if mw < 160:
        return "small"
    if mw < 300:
        return "medium"
    if mw < 500:
        return "large"
    return "very_large"

def _check_lipinski(mw, logp, hbd, hba):
    if any(_is_missing(v) for v in [mw, logp, hbd, hba]):
        return None, None
    violations = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
    return violations == 0, violations

def _check_veber(tpsa, rotatable_bonds):
    if any(_is_missing(v) for v in [tpsa, rotatable_bonds]):
        return None
    return tpsa <= 140 and rotatable_bonds <= 10

def _check_egan(logp, tpsa):
    if any(_is_missing(v) for v in [logp, tpsa]):
        return None
    return logp <= 5.88 and tpsa <= 131

def _check_ghose(mw, logp, molar_refractivity):
    if any(_is_missing(v) for v in [mw, logp, molar_refractivity]):
        return None
    return 160 <= mw <= 480 and -0.4 <= logp <= 5.6 and 40 <= molar_refractivity <= 130

def _check_pains(mol):
    pains_patterns = {
        'thiazole': '[#16]~[#6]~[#7]~[#6]~[#16]',
        'catechol': '[#6]~[#6]([#8])[#6]~[#8]',
        'quinone': '[#6](=[#8])[#6]~[#6](=[#8])',
        'Michael_acceptor': '[#6](=[#8])[#6]=[#6]',
        'cyanamide': '[#7][#6]#[#7]',
        'acrylamide': '[#6]=[#6][#6](=[#8])[#7]',
    }
    alerts = []
    for name, smarts in pains_patterns.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            alerts.append(name)
    return len(alerts) > 0, alerts

def _linear_decrease(value, low, high):
    if value <= low:
        return 1.0
    if value > high:
        return 0.0
    return (high - value) / (high - low)

def _score_clogp(value):
    return _linear_decrease(value, 3.0, 5.0)

def _score_clogd(value):
    return _linear_decrease(value, 2.0, 4.0)

def _score_mw(value):
    return _linear_decrease(value, 360.0, 500.0)

def _score_tpsa(value):
    if value <= 20.0:
        return 0.0
    if value <= 40.0:
        return (value - 20.0) / 20.0
    if value <= 90.0:
        return 1.0
    if value <= 120.0:
        return (120.0 - value) / 30.0
    return 0.0

def _score_hbd(value):
    return _linear_decrease(value, 0.5, 3.5)

def _score_pka(value):
    return _linear_decrease(value, 8.0, 10.0)

def _table_from_cxcalc_output(output):
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    lines = [line for line in lines if not line.lower().startswith(("warning", "error"))]
    if len(lines) < 2:
        return None, None
    header = re.split(r"\t+|\s{2,}", lines[0].strip())
    data = re.split(r"\t+|\s{2,}", lines[1].strip())
    return header, data

def _parse_cxcalc_number(output, accepted_header_fragments):
    header, data = _table_from_cxcalc_output(output)
    if not header or not data:
        return None
    for index, name in enumerate(header):
        normalized = name.lower()
        if any(fragment in normalized for fragment in accepted_header_fragments) and index < len(data):
            try:
                return float(data[index])
            except (TypeError, ValueError):
                pass
    for token in data[1:]:
        if FLOAT_RE.fullmatch(token.strip()):
            return float(token)
    return None

def _calculate_logd_and_pka(smiles):
    try:
        logd_output = subprocess.run(
            ["cxcalc", "logd", "pH=7.4", smiles],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        pka_output = subprocess.run(
            ["cxcalc", "pka", smiles],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    if logd_output.returncode != 0 or pka_output.returncode != 0:
        return None, None
    logd = _parse_cxcalc_number(logd_output.stdout, ("logd",))
    pka = _parse_cxcalc_number(pka_output.stdout, ("bpka", "basic"))
    return logd, pka

def _calculate_cns_mpo(mw, logp, tpsa, hbd, logd=None, pka=None):
    """Calculate CNS MPO, falling back to LogP and neutral pKa when cxcalc is unavailable."""
    logd = logp if logd is None else logd
    pka = 8.0 if pka is None else pka
    return (
        _score_clogp(logp)
        + _score_clogd(logd)
        + _score_mw(mw)
        + _score_tpsa(tpsa)
        + _score_hbd(hbd)
        + _score_pka(pka)
    )

def calculate_molecular_properties(mol):
    """Calculate molecular properties safely"""
    try:
        tpsa = Descriptors.TPSA(mol)
        logp = Crippen.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        ring_count = rdMolDescriptors.CalcNumRings(mol)
        molar_refractivity = Crippen.MolMR(mol)
        lipinski_pass, lipinski_violations = _check_lipinski(mw, logp, hbd, hba)
        veber_pass = _check_veber(tpsa, rotatable_bonds)
        egan_pass = _check_egan(logp, tpsa)
        ghose_pass = _check_ghose(mw, logp, molar_refractivity)
        pains_flag, pains_alerts = _check_pains(mol)
        aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms())
        heterocycle_present = any(
            atom.IsInRing() and atom.GetAtomicNum() not in (1, 6)
            for atom in mol.GetAtoms()
        )
        peptide_like = bool(mol.HasSubstructMatch(Chem.MolFromSmarts('[#6](=[#8])[#7]')))
        num_c = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
        num_o = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
        lipid_like = num_c > 15 and num_o <= num_c / 3
        smiles = Chem.MolToSmiles(mol, canonical=True)
        logd, pka = _calculate_logd_and_pka(smiles)
        cns_mpo = _calculate_cns_mpo(mw, logp, tpsa, hbd, logd=logd, pka=pka)
        logd = logp if logd is None else logd

        profile = {
            "physicochemical": {
                "tpsa": float(tpsa),
                "polarity_bin": _get_polarity_bin(tpsa),
                "logp": float(logp),
                "lipophilicity_bin": _get_lipophilicity_bin(logp),
                "logd": float(logd),
                "cns_mpo": float(cns_mpo),
            },
            "structural": {
                "size_bin": _get_size_bin(mw),
                "aromatic": bool(aromatic),
                "ring_count": int(ring_count),
                "heterocycle_present": bool(heterocycle_present),
                "peptide_like": bool(peptide_like),
                "lipid_like": bool(lipid_like),
            },
            "drug_candidacy": {
                "lipinski": {"pass": lipinski_pass, "violations": lipinski_violations},
                "veber": {"pass": veber_pass},
                "egan": {"pass": egan_pass},
                "ghose": {"pass": ghose_pass},
                "pains": {"flag": pains_flag, "alerts": pains_alerts},
            },
        }

        return {
            'mw': mw,
            'logp': logp,
            'hbd': hbd,
            'hba': hba,
            'tpsa': tpsa,
            'rotatable_bonds': rotatable_bonds,
            'heavy_atoms': Descriptors.HeavyAtomCount(mol),
            'ring_count': ring_count,
            'heterocycle_present': heterocycle_present,
            'molar_refractivity': molar_refractivity,
            'peptide_like': peptide_like,
            'lipid_like': lipid_like,
            'polarity_bin': _get_polarity_bin(tpsa),
            'lipophilicity_bin': _get_lipophilicity_bin(logp),
            'size_bin': _get_size_bin(mw),
            'tpsa_bin': _get_polarity_bin(tpsa),
            'logp_bin': _get_lipophilicity_bin(logp),
            'mw_bin': _get_size_bin(mw),
            'logd': logd,
            'cns_mpo': cns_mpo,
            'aromatic': aromatic,
            'lipinski_pass': lipinski_pass,
            'lipinski_violations': lipinski_violations,
            'veber_pass': veber_pass,
            'egan_pass': egan_pass,
            'ghose_pass': ghose_pass,
            'pains_flag': pains_flag,
            'pains_alerts': pains_alerts,
            'profile_json': json.dumps(profile),
        }
    except Exception as e:
        logger.error(f"Failed to calculate properties: {str(e)}")
        return None

def process_batch_molecules(input_data, input_type, models):
    """Process multiple molecules for batch prediction"""
    results = []
    
    if input_type == "csv":
    # Expected CSV columns: either 'smiles'/'Smiles' or 'name'/'Name' or both
    # Check for smiles column (case-insensitive)
        smiles_col = None
        name_col = None
    
        for col in input_data.columns:
            if col.lower() == 'smiles':
                smiles_col = col
            elif col.lower() == 'name':
                name_col = col
    
        if smiles_col:
            molecules = input_data[smiles_col].tolist()
            if name_col:
                names = input_data[name_col].tolist()
            else:
                names = molecules
            input_method = 'smiles'
        elif name_col:
            molecules = input_data[name_col].tolist()
            names = molecules
            input_method = 'name'
    
        else:
            return [], "CSV must contain either 'smiles'/'Smiles' or 'name'/'Name' column"
    else:
        # Text input - one per line
        lines = input_data.strip().split('\n')
        molecules = [line.strip() for line in lines if line.strip()]
        names = molecules
        input_method = 'mixed'  # Could be names or SMILES
    
    # Extract all valid SMILES first
    valid_smiles = []
    valid_indices = []
    molecule_info = []  # Store associated info
    
    for i, molecule in enumerate(molecules):
        try:
            mol = None
            smiles = None
            actual_name = names[i] if i < len(names) else molecule
            
            # Try as SMILES first
            if input_method in ['smiles', 'mixed']:
                mol = Chem.MolFromSmiles(molecule)
                if mol:
                    smiles = molecule
            
            # If not SMILES, try as name
            if not mol and input_method in ['name', 'mixed']:
                try:
                    smiles = get_smiles(molecule)
                    mol = Chem.MolFromSmiles(smiles) if smiles else None
                except:
                    pass
            
            if mol and smiles:
                valid_smiles.append(smiles)
                valid_indices.append(i)
                molecule_info.append({
                    'mol': mol,
                    'smiles': smiles,
                    'input': molecule,
                    'name': actual_name
                })
            else:
                # Add error result immediately
                results.append({
                    'chembl_id': None,
                    'mol': None,
                    'input': molecule,
                    'name': actual_name,
                    'smiles': None,
                    'formula': None,
                    'status': 'Error',
                    'error': 'Could not process molecule',
                    'prediction': None,
                    'confidence': None,
                    'uncertainty': None,
                    'agreement': None
                })
        except Exception as e:
            results.append({
                'chembl_id': None,
                'mol': None,
                'input': molecule,
                'name': names[i] if i < len(names) else molecule,
                'smiles': None,
                'formula': None,
                'status': 'Error',
                'error': str(e),
                'prediction': None,
                'confidence': None,
                'uncertainty': None,
                'agreement': None
            })
    
    # Batch calculate strict-validation feature views for all valid SMILES.
    if valid_smiles:
        try:
            feature_views, failed_indices, embedding_error = _build_webapp_feature_views(valid_smiles)
            
            # Make predictions for each molecule
            for idx, (smiles, info) in enumerate(zip(valid_smiles, molecule_info)):
                if idx in failed_indices:
                    results.append({
                        'chembl_id': None,
                        'mol': info['mol'],
                        'input': info['input'],
                        'name': info['name'],
                        'smiles': smiles,
                        'formula': get_formula(smiles),
                        'status': 'Error',
                        'error': 'Descriptor calculation failed',
                        'prediction': None,
                        'confidence': None,
                        'uncertainty': None,
                        'agreement': None
                    })
                    continue
                
                try:
                    prediction_result, model_errors = _predict_from_feature_views(feature_views, idx, models)
                    if prediction_result is None:
                        raise ValueError(f"All models failed: {model_errors}")
                    if embedding_error and "PaDEL+Morgan+Embeddings XGBoost" in models:
                        prediction_result["model_errors"]["PaDEL+Morgan+Embeddings XGBoost"] = embedding_error
                    
                    # Get ChEMBL info
                    chembl_info = get_chembl_info(smiles)
                    formula = get_formula(smiles)
                    properties = calculate_molecular_properties(info['mol'])
                    
                    result = {
                        'chembl_id': chembl_info.get('ChEMBL ID') if chembl_info else None,
                        'mol': info['mol'],
                        'input': info['input'],
                        'name': chembl_info.get('Name') if chembl_info else info['name'],
                        'smiles': smiles,
                        'formula': formula,
                        'status': 'Success',
                        'error': None,
                        'prediction': prediction_result["ensemble_pred"],
                        'confidence': prediction_result["avg_confidence"],
                        'uncertainty': prediction_result["uncertainty"],
                        'agreement': prediction_result["agreement"],
                        'num_models': prediction_result["num_models"],
                        'individual_predictions': prediction_result["predictions"],
                        'individual_confidences': prediction_result["confidences"],
                        'positive_probabilities': prediction_result["positive_probabilities"],
                        'model_errors': prediction_result["model_errors"],
                    }
                    
                    if properties:
                        result.update({
                            'molecular_weight': properties['mw'],
                            'logp': properties['logp'],
                            'hbd': properties['hbd'],
                            'hba': properties['hba'],
                            'tpsa': properties['tpsa'],
                            'rotatable_bonds': properties['rotatable_bonds'],
                            'heavy_atoms': properties['heavy_atoms'],
                            'ring_count': properties['ring_count'],
                            'molar_refractivity': properties['molar_refractivity'],
                            'heterocycle_present': properties['heterocycle_present'],
                            'peptide_like': properties['peptide_like'],
                            'lipid_like': properties['lipid_like'],
                            'aromatic': properties['aromatic'],
                            'tpsa_bin': properties['tpsa_bin'],
                            'logp_bin': properties['logp_bin'],
                            'mw_bin': properties['mw_bin'],
                            'logd': properties['logd'],
                            'cns_mpo': properties['cns_mpo'],
                            'lipinski_pass': properties['lipinski_pass'],
                            'veber_pass': properties['veber_pass'],
                            'egan_pass': properties['egan_pass'],
                            'ghose_pass': properties['ghose_pass'],
                            'pains_flag': properties['pains_flag'],
                            'profile_json': properties['profile_json'],
                        })
                    
                    results.append(result)
                    
                except Exception as e:
                    results.append({
                        'chembl_id': None,
                        'mol': info['mol'],
                        'input': info['input'],
                        'name': info['name'],
                        'smiles': smiles,
                        'formula': get_formula(smiles),
                        'status': 'Error',
                        'error': f'Prediction failed: {str(e)}',
                        'prediction': None,
                        'confidence': None,
                        'uncertainty': None,
                        'agreement': None
                    })
        
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            return [], f"Batch descriptor calculation failed: {str(e)}"
    
    return results, None

def scale_descriptors(input_df, models):
    """
    Scale input descriptor DataFrame for legacy standalone models.
    The hypertuned 80/20 models loaded by the app are saved sklearn/imblearn
    pipelines, so their imputer/scaler/model steps run inside predict().
    In that case no external scaler is loaded and descriptors are returned raw.
    """
    scaler = models.get('scaler', None)
    if scaler is not None:
        scaled = scaler.transform(input_df)
        return pd.DataFrame(scaled, columns=input_df.columns, index=input_df.index)
    else:
        logger.warning("WARNING!!! - No scaler found in models dict. Returning unscaled descriptors.")
        return input_df

def predict_bbb_padel(smiles, models):
    """
    Predict BBB penetration using the strict-validation model set.
    Returns a dict of predictions and confidences for each model.
    """
    try:
        feature_views, failed_indices, embedding_error = _build_webapp_feature_views([smiles])
        if 0 in failed_indices:
            return None, None, None, None, "PaDEL descriptor calculation failed"
        prediction_result, model_errors = _predict_from_feature_views(feature_views, 0, models)
        if prediction_result is None:
            return None, None, None, None, f"All models failed to make predictions: {model_errors}"
        if embedding_error and "PaDEL+Morgan+Embeddings XGBoost" in models:
            prediction_result["model_errors"]["PaDEL+Morgan+Embeddings XGBoost"] = embedding_error
        return (
            prediction_result["predictions"],
            prediction_result["confidences"],
            prediction_result["ensemble_pred"],
            prediction_result["avg_confidence"],
            None,
        )
    except Exception as e:
        error_msg = f"PaDEL prediction failed: {str(e)}"
        logger.error(error_msg)
        return None, None, None, None, error_msg
