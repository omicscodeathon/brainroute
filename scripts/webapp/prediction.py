import pandas as pd
import numpy as np
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from config import DEFAULT_MODEL
import logging

from api import get_chembl_info, get_formula

logger = logging.getLogger(__name__)

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
    
    for i, molecule in enumerate(molecules):
        try:
            # Determine if it's SMILES or name
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
                from scripts.webapp.api import get_smiles
                try:
                    smiles = get_smiles(molecule)
                    mol = Chem.MolFromSmiles(smiles) if smiles else None
                except:
                    pass
            
            if mol and smiles:
                # Make prediction with uncertainty
                pred_result, error = predict_bbb_penetration_with_uncertainty(mol, models)
                info = get_chembl_info(smiles)
                formula = get_formula(smiles)
                if error:
                    results.append({
                        'chembl_id': info.get('ChEMBL ID') if info else None,
                        'mol': mol,
                        'input': molecule,
                        'name': info.get('Name') if info else actual_name if actual_name else 'Unknown',
                        'smiles': smiles,
                        'formula': formula,
                        'status': 'Error',
                        'error': error,
                        'prediction': None,
                        'confidence': None,
                        'uncertainty': None,
                        'agreement': None
                    })
                else:
                    # Calculate properties
                    properties = calculate_molecular_properties(mol)
                    result = {
                        'chembl_id': info.get('ChEMBL ID') if info else None,
                        'mol': mol,
                        'input': molecule,
                        'name': info.get('Name') if info else actual_name if actual_name else 'Unknown',
                        'smiles': smiles,
                        'formula': formula,
                        'status': 'Success',
                        'error': None,
                        'prediction': pred_result['prediction'],
                        'confidence': pred_result['confidence'],
                        'uncertainty': pred_result['uncertainty'],
                        'agreement': pred_result['agreement'],
                        'num_models': pred_result['num_models']
                    }
                    
                    # Add molecular properties
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
            else:
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
    
    return results, None