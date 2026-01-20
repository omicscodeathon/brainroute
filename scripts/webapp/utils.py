# utils.py
import streamlit as st
import time
import re
import urllib.parse
import pandas as pd
import io
import base64
from datetime import datetime
import json
from openai import OpenAI
from config import (MODEL_PATHS, AI_MODEL_NAME, API_GENERATION_CONFIG, PROMPT_TEMPLATES, 
                   USE_HF_INFERENCE_API, HF_API_TOKEN)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource(show_spinner=False)
def load_ml_models():
    """Load ML models once and cache them with proper error handling, including feature names."""
    models = {}
    errors = []
    import joblib
    import os
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_models = len(MODEL_PATHS)
    for i, (model_name, model_path) in enumerate(MODEL_PATHS.items()):
        try:
            status_text.text(f"Loading {model_name} model...")
            models[model_name] = joblib.load(model_path)
            progress_bar.progress((i + 1) / total_models)
            logger.info(f"Successfully loaded {model_name} model")
        except Exception as e:
            error_msg = f"Failed to load {model_name}: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
    # Load feature names
    feature_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../output/models/feature_names.pkl'))
    try:
        models['feature_names'] = joblib.load(feature_path)
        logger.info(f"Loaded feature names from {feature_path}")
    except Exception as e:
        error_msg = f"Could not load feature names: {e}"
        errors.append(error_msg)
        logger.error(error_msg)
    progress_bar.empty()
    status_text.empty()
    return models, errors

def check_hf_api_status():
    """Check if Hugging Face API is available using OpenAI client"""
    try:
        if not HF_API_TOKEN:
            return False
            
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=HF_API_TOKEN,
        )

        # Simple test query
        completion = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct:novita",
            messages=[
                {
                    "role": "user",
                    "content": "Hello"
                }
            ],
            max_tokens=10,
            timeout=10
        )
        
        return completion.choices[0].message.content is not None
        
    except Exception as e:
        logger.error(f"API status check failed: {str(e)}")
        return False

def generate_with_hf_api(prompt, max_retries=3):
    """Generate text using OpenAI client with Hugging Face router"""
    if not HF_API_TOKEN:
        return None, "Hugging Face API token not found. Please set HUGGINGFACE_API_TOKEN environment variable."
    
    try:
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=HF_API_TOKEN,
        )
        
        # Create system message for pharmacology expertise
        system_message = """You are a helpful assistant specializing in pharmacology and drug discovery. 
        Provide accurate, concise information about molecules and their blood-brain barrier penetration properties. 
        Keep responses focused and under 250 words."""
        
        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model="meta-llama/Meta-Llama-3-8B-Instruct:novita",
                    messages=[
                        {
                            "role": "system",
                            "content": system_message
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=API_GENERATION_CONFIG.get('max_new_tokens', 250),
                    temperature=API_GENERATION_CONFIG.get('temperature', 0.7),
                    top_p=API_GENERATION_CONFIG.get('top_p', 0.9),
                    timeout=30
                )
                
                if completion.choices and len(completion.choices) > 0:
                    generated_text = completion.choices[0].message.content
                    
                    if generated_text:
                        # Clean up the generated text
                        cleaned_text = clean_generated_text(generated_text)
                        complete_text = ensure_complete_sentence(cleaned_text)
                        
                        return complete_text, None
                    else:
                        return None, "Empty response from API"
                else:
                    return None, "No response choices returned"
                    
            except Exception as e:
                error_msg = str(e)
                
                # Handle specific error types
                if "rate limit" in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = 5 + (attempt * 2)
                        time.sleep(wait_time)
                        continue
                    else:
                        return None, "Rate limit exceeded. Please try again later."
                
                elif "timeout" in error_msg.lower():
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    else:
                        return None, "Request timed out. Please try again."
                
                elif "model is currently loading" in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = 10 + (attempt * 5)
                        time.sleep(wait_time)
                        continue
                    else:
                        return None, "Model is currently loading. Please try again in a few minutes."
                
                else:
                    return None, f"API request failed: {error_msg}"
        
        return None, "Failed after multiple attempts"
        
    except Exception as e:
        return None, f"Failed to initialize OpenAI client: {str(e)}"

@st.cache_data(ttl=300)  # Cache for 5 minutes
def generate_ai_response_cached(prompt):
    """Cached wrapper for AI response generation"""
    return generate_with_hf_api(prompt)

def load_ai_model():
    """Check if OpenAI client can connect to HF API"""
    try:
        if not USE_HF_INFERENCE_API:
            return None, None, "Local model loading not supported in this version"
        
        if not HF_API_TOKEN:
            return None, None, "Please set HUGGINGFACE_API_TOKEN environment variable"
        
        # Test API connectivity
        if check_hf_api_status():
            logger.info(f"Successfully connected to HF API for {AI_MODEL_NAME}")
            return "api_ready", "api_ready", None
        else:
            return None, None, "Cannot connect to Hugging Face API. Please check your token and try again."
    
    except Exception as e:
        error_msg = f"Failed to initialize AI model: {str(e)}"
        logger.error(error_msg)
        return None, None, error_msg

def ensure_complete_sentence(text):
    """Ensures the text ends with a complete sentence"""
    if not text.strip():
        return text
    
    sentence_endings = ['.', '!', '?']
    last_ending_pos = -1
    
    for ending in sentence_endings:
        pos = text.rfind(ending)
        if pos > last_ending_pos:
            last_ending_pos = pos
    
    if last_ending_pos == -1:
        return text.strip() + "."
    else:
        return text[:last_ending_pos + 1].strip()

def clean_generated_text(text):
    """Clean up generated text artifacts"""
    if not text:
        return ""
    
    # Remove common artifacts and formatting
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.{2,}', '.', text)
    
    # Remove common research paper artifacts
    text = re.sub(r'ABSTRACT.*?WORDS\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'LEARNING OBJECTIVE.*?:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'DATA SOURCES.*?:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'CONCLUSIONS?:', '', text, flags=re.IGNORECASE)
    
    # Remove incomplete sentences at the end
    sentences = text.split('.')
    if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
        text = '.'.join(sentences[:-1]) + '.'
    
    return text.strip()

def create_chatgpt_link(molecule_name):
    """Creates a ChatGPT link with pre-populated prompt"""
    prompt = f"Tell me more about {molecule_name} and its relevance in CNS drug discovery, with context of blood-brain barrier (BBB) penetration."
    encoded_prompt = urllib.parse.quote(prompt)
    return f"https://chat.openai.com/?q={encoded_prompt}"

def generate_ai_response(tokenizer, model, prompt):
    """Generate AI response using OpenAI client with HF router"""
    if USE_HF_INFERENCE_API:
        return generate_ai_response_cached(prompt)
    else:
        return None, "Local model generation not supported"

def create_download_link(df, filename, file_format="csv"):
    """Create a download link for DataFrame"""
    if file_format == "csv":
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">📥 Download CSV</a>'
    elif file_format == "excel":
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='BBB_Predictions', index=False)
        excel_data = output.getvalue()
        b64 = base64.b64encode(excel_data).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">📥 Download Excel</a>'
    elif file_format == "json":
        json_data = df.to_json(orient='records', indent=2)
        b64 = base64.b64encode(json_data.encode()).decode()
        href = f'<a href="data:file/json;base64,{b64}" download="{filename}.json">📥 Download JSON</a>'
    
    return href

def format_batch_results_for_display(results):
    """Format batch results for display in Streamlit"""
    if not results:
        return pd.DataFrame()
    
    # Create a clean dataframe for display
    display_data = []
    for result in results:
        display_row = {
            'Input': result.get('input', ''),
            'Name': result.get('name', ''),
            'Status': result.get('status', 'Unknown'),
            'Prediction': result.get('prediction', 'N/A'),
            'Confidence (%)': f"{result.get('confidence', 0):.1f}" if result.get('confidence') else 'N/A',
            'Uncertainty (%)': f"{result.get('uncertainty', 0):.1f}" if result.get('uncertainty') else 'N/A',
            'Agreement (%)': f"{result.get('agreement', 0):.1f}" if result.get('agreement') else 'N/A',
            'MW': f"{result.get('molecular_weight', 0):.1f}" if result.get('molecular_weight') else 'N/A',
            'LogP': f"{result.get('logp', 0):.2f}" if result.get('logp') else 'N/A',
            'SMILES': result.get('smiles', 'N/A')
        }
        
        if result.get('status') == 'Error':
            display_row['Error'] = result.get('error', 'Unknown error')
        
        display_data.append(display_row)
    
    return pd.DataFrame(display_data)

def create_summary_stats(results):
    """Create summary statistics from batch results"""
    if not results:
        return {}
    
    successful = [r for r in results if r.get('status') == 'Success']
    failed = [r for r in results if r.get('status') == 'Error']
    
    bbb_positive = [r for r in successful if r.get('prediction') == 'BBB+']
    bbb_negative = [r for r in successful if r.get('prediction') == 'BBB-']
    
    stats = {
        'total_molecules': len(results),
        'successful_predictions': len(successful),
        'failed_predictions': len(failed),
        'success_rate': (len(successful) / len(results)) * 100 if results else 0,
        'bbb_positive_count': len(bbb_positive),
        'bbb_negative_count': len(bbb_negative),
        'bbb_positive_rate': (len(bbb_positive) / len(successful)) * 100 if successful else 0,
        'avg_confidence': sum(r.get('confidence', 0) for r in successful) / len(successful) if successful else 0,
        'avg_uncertainty': sum(r.get('uncertainty', 0) for r in successful) / len(successful) if successful else 0,
        'avg_agreement': sum(r.get('agreement', 0) for r in successful) / len(successful) if successful else 0
    }
    
    return stats

# def calculate_padel_descriptors(smiles, padel_jar_path='padel.sh', descriptors_output='padel_temp.csv'):
#     """
#     Calculate PaDEL descriptors for a given SMILES string using the PaDEL-Descriptor tool.
#     Returns a pandas DataFrame with descriptor values for the molecule.
#     """
#     import subprocess
#     import tempfile
#     import os
#     import pandas as pd

#     # Write SMILES to a temporary file
#     with tempfile.NamedTemporaryFile(mode='w', suffix='.smi', delete=False) as smi_file:
#         smi_file.write(smiles + '\tMol1\n')
#         smi_path = smi_file.name

#     # Prepare output file
#     with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as out_file:
#         out_path = out_file.name

#     # Run PaDEL-Descriptor via shell script (padel.sh should call the JAR with correct args)
#     try:
#         subprocess.run([padel_jar_path, '-removesalt', '-standardizenitro', '-fingerprints', '-descriptortypes', '', '-dir', smi_path, '-file', out_path], check=True)
#         df = pd.read_csv(out_path)
#         # Remove temp files
#         os.remove(smi_path)
#         os.remove(out_path)
#         return df
#     except Exception as e:
#         # Clean up temp files
#         if os.path.exists(smi_path):
#             os.remove(smi_path)
#         if os.path.exists(out_path):
#             os.remove(out_path)
#         raise RuntimeError(f"PaDEL descriptor calculation failed: {e}")

def calculate_padel_descriptors(smiles):
    """
    Calculate PaDEL descriptors for a given SMILES string using padelpy.
    Returns a pandas DataFrame with descriptor values.
    """
    try:
        from padelpy import from_smiles
        import pandas as pd
        
        # Calculate descriptors directly from SMILES
        # This returns a pandas Series
        descriptors = from_smiles(smiles, fingerprints=False, descriptors=True)
        
        # Convert to DataFrame (single row)
        df = pd.DataFrame([descriptors])
        
        return df
        
    except Exception as e:
        logger.error(f"PaDEL descriptor calculation failed for SMILES {smiles}: {str(e)}")
        raise RuntimeError(f"PaDEL descriptor calculation failed: {e}")


def calculate_padel_descriptors_batch(smiles_list):
    """
    Calculate PaDEL descriptors for multiple SMILES strings efficiently.
    Returns a pandas DataFrame with descriptor values for all molecules.
    """
    try:
        from padelpy import from_smiles
        import pandas as pd
        
        all_descriptors = []
        failed_indices = []
        
        for idx, smiles in enumerate(smiles_list):
            try:
                descriptors = from_smiles(smiles, fingerprints=False, descriptors=True)
                all_descriptors.append(descriptors)
            except Exception as e:
                logger.warning(f"Failed to calculate descriptors for SMILES at index {idx}: {e}")
                failed_indices.append(idx)
                all_descriptors.append(None)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_descriptors)
        
        return df, failed_indices
        
    except Exception as e:
        logger.error(f"Batch PaDEL descriptor calculation failed: {str(e)}")
        raise RuntimeError(f"Batch descriptor calculation failed: {e}")

def safe_align_features(input_df, expected_features, molecule_name="Unknown"):
    """
    Aligns the input descriptor DataFrame to match the expected features for model prediction.
    Returns (aligned_df, error_message). If alignment fails or <90% overlap, returns (None, error_message).
    """
    import numpy as np
    import logging
    logger = logging.getLogger("safe_align_features")
    
    # Calculate overlap
    input_features = set(input_df.columns)
    expected_features_set = set(expected_features)
    overlap = input_features & expected_features_set
    overlap_pct = len(overlap) / len(expected_features_set) * 100 if expected_features_set else 0
    missing = expected_features_set - input_features
    extra = input_features - expected_features_set
    
    if overlap_pct < 90:
        error = (f"Critical feature mismatch for {molecule_name}: "
                 f"{len(overlap)}/{len(expected_features_set)} features present ({overlap_pct:.1f}%). "
                 f"Missing: {sorted(list(missing))[:5]}... (total {len(missing)})")
        logger.error(error)
        return None, error
    if missing:
        logger.warning(f"{len(missing)} features missing for {molecule_name}. Filling with 0. Examples: {sorted(list(missing))[:5]}")
    if extra:
        logger.warning(f"{len(extra)} extra features in input for {molecule_name}. Examples: {sorted(list(extra))[:5]}")
    # Align and fill missing with 0
    aligned_df = input_df.reindex(columns=expected_features, fill_value=0)
    return aligned_df, None






