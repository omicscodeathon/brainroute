import pandas as pd
import numpy as np
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from config import DEFAULT_MODEL
import logging
from sklearn.preprocessing import StandardScaler

from api import get_chembl_info, get_formula, get_smiles
from utils import calculate_padel_descriptors, calculate_padel_descriptors_batch

logger = logging.getLogger(__name__)

# PADEL_JAR_PATH = '../../notebooks/padel.sh'  # Adjust if needed

def predict_bbb_penetration_with_uncertainty(mol, models):
    """Predict BBB penetration with uncertainty quantification using multiple models"""
    try:
        # Calculate descriptors
        descr = pd.DataFrame([Descriptors.CalcMolDescriptors(mol)])
        
        # Scale descriptors
        if 'scaler' not in models:
            raise ValueError("Scaler not found in loaded models")
        
        scaler = models['scaler']
        descr_scaled = scaler.transform(descr)
        
        # Get predictions from all available models
        predictions = {}
        confidences = {}
        
        for model_name in ['KNN', 'XGB', 'SVM', 'RF', 'LR']:  # Add more models as available
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

def calculate_molecular_properties(mol):
    """Calculate molecular properties safely"""
    try:
        return {
            'mw': Descriptors.MolWt(mol),
            'logp': Crippen.MolLogP(mol),
            'hbd': Lipinski.NumHDonors(mol),
            'hba': Lipinski.NumHAcceptors(mol),
            'tpsa': Descriptors.TPSA(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'heavy_atoms': Descriptors.HeavyAtomCount(mol)
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
    
    # Batch calculate PaDEL descriptors for all valid SMILES
    if valid_smiles:
        try:
            padel_df, failed_indices = calculate_padel_descriptors_batch(valid_smiles)
            # Ensure all columns are numeric
            padel_df = padel_df.apply(pd.to_numeric, errors='coerce').fillna(0) #converts to int - if NaN then 0 
            expected_features = models.get('feature_names')
            if expected_features:
                from utils import safe_align_features
                padel_df, align_error = safe_align_features(padel_df, expected_features, "batch")
                if align_error:
                    return [], f"Feature alignment failed: {align_error}"
            # Scale batch descriptors
            padel_df = scale_descriptors(padel_df, models)
            
            # Make predictions for each molecule
            for idx, (smiles, info) in enumerate(zip(valid_smiles, molecule_info)):
                if idx in failed_indices or padel_df.iloc[idx].isna().all():
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
                    # Get single row for this molecule
                    single_padel = padel_df.iloc[[idx]].drop(columns=['Name'], errors='ignore')
                    
                    # Make predictions with all models
                    predictions = {}
                    confidences = {}
                    
                    for model_name in ['KNN', 'LGBM', 'ET']:
                        if model_name in models:
                            try:
                                model = models[model_name]
                                pred = model.predict(single_padel)
                                
                                if hasattr(model, 'predict_proba'):
                                    conf = model.predict_proba(single_padel)
                                    confidence = conf[0][1] * 100 if conf.shape[1] > 1 else conf[0][0] * 100
                                else:
                                    confidence = None
                                
                                predictions[model_name] = int(pred[0])
                                confidences[model_name] = confidence
                            except Exception as e:
                                logger.warning(f"Model {model_name} failed for molecule {idx}: {str(e)}")
                                continue
                    
                    if not predictions:
                        raise ValueError("All models failed")
                    
                    # Calculate ensemble prediction
                    pred_values = list(predictions.values())
                    avg_pred = sum(pred_values) / len(pred_values)
                    ensemble_pred = "BBB+" if avg_pred >= 0.5 else "BBB-"
                    
                    # Calculate average confidence
                    valid_confs = [c for c in confidences.values() if c is not None]
                    avg_confidence = sum(valid_confs) / len(valid_confs) if valid_confs else 50.0
                    
                    # Calculate agreement
                    agreement = (sum(pred_values) / len(pred_values)) * 100
                    if agreement < 50:
                        agreement = 100 - agreement
                    
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
                        'prediction': ensemble_pred,
                        'confidence': avg_confidence,
                        'uncertainty': 0,  # You can calculate this from model disagreement
                        'agreement': agreement,
                        'num_models': len(predictions)
                    }
                    
                    if properties:
                        result.update({
                            'molecular_weight': properties['mw'],
                            'logp': properties['logp'],
                            'hbd': properties['hbd'],
                            'hba': properties['hba'],
                            'tpsa': properties['tpsa'],
                            'rotatable_bonds': properties['rotatable_bonds'],
                            'heavy_atoms': properties['heavy_atoms']
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
    Scale input descriptor DataFrame using the StandardScaler saved in models dict.
    Returns scaled DataFrame. If scaler not found, returns input_df unchanged.
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
    Predict BBB penetration using PaDEL descriptors and the provided models (KNN, LGBM, ET).
    Returns a dict of predictions and confidences for each model.
    """
    try:
        # Calculate PaDEL descriptors using padelpy
        padel_df = calculate_padel_descriptors(smiles)
        padel_df = padel_df.drop(columns=['Name'], errors='ignore')
        # Ensure all columns are numeric
        padel_df = padel_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        expected_features = models.get('feature_names')
        if expected_features:
            from utils import safe_align_features
            padel_df, align_error = safe_align_features(padel_df, expected_features, smiles[:20])
            if align_error:
                return None, None, align_error
        # Scale descriptors using StandardScaler
        padel_df = scale_descriptors(padel_df, models)
        predictions = {}
        confidences = {}
        for model_name in ['KNN', 'LGBM', 'ET']:
            if model_name in models:
                try:
                    model = models[model_name]
                    pred = model.predict(padel_df)
                    if hasattr(model, 'predict_proba'):
                        conf = model.predict_proba(padel_df)
                        confidence = conf[0].max() * 100
                    else:
                        confidence = None
                    predictions[model_name] = int(pred[0])
                    confidences[model_name] = confidence
                except Exception as e:
                    logger.warning(f"Model {model_name} prediction failed: {str(e)}")
                    continue
        if not predictions:
            return None, None, "All models failed to make predictions"
        return predictions, confidences, None
    except Exception as e:
        error_msg = f"PaDEL prediction failed: {str(e)}"
        logger.error(error_msg)
        return None, None, error_msg