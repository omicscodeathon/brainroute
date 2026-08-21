import os

import streamlit as st

# Model Configuration
MODEL_PATHS = {
    'PaDEL+Morgan LGBM': '../../brainroute_ml_validation/models/full_refit/full_refit__padel_morgan__lightgbm.joblib',
    'PaDEL+Morgan Extra Trees': '../../brainroute_ml_validation/models/full_refit/full_refit__padel_morgan__extra_trees.joblib',
    'PaDEL+Morgan+Embeddings XGBoost': '../../brainroute_ml_validation/models/full_refit/full_refit__padel_morgan_embeddings__xgboost.joblib',
}
MODEL_FEATURE_VIEWS = {
    'PaDEL+Morgan LGBM': 'padel_morgan',
    'PaDEL+Morgan Extra Trees': 'padel_morgan',
    'PaDEL+Morgan+Embeddings XGBoost': 'padel_morgan_embeddings',
}
VALIDATION_CONFIG_PATH = '../../brainroute_ml_validation/configs/validation_config.yaml'
FEATURE_NAMES_PATH = None

# AI Model Configuration - Using OpenAI client with HF router
AI_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
USE_HF_INFERENCE_API = True
DEFAULT_MODEL = "PaDEL+Morgan LGBM"

# Hugging Face API Configuration
# HF_TOKEN is the canonical Streamlit secret. Keep the former
# HUGGINGFACE_API_TOKEN name as a deployment-compatible fallback.
try:
    _HF_SECRET_TOKEN = (
        st.secrets.get("HF_TOKEN", "")
        or st.secrets.get("HUGGINGFACE_API_TOKEN", "")
    )
except Exception:
    _HF_SECRET_TOKEN = ""

HF_API_TOKEN = (
    _HF_SECRET_TOKEN
    or os.getenv("HF_TOKEN", "")
    or os.getenv("HUGGINGFACE_API_TOKEN", "")
)

# Generation Parameters for OpenAI client
API_GENERATION_CONFIG = {
    'max_new_tokens': 500,
    'temperature': 0.7,
    'top_p': 0.9,
}

# UI Configuration
PAGE_CONFIG = {
    'page_title': "BrainRoute",
    'page_icon': "🧪",
    'layout': "wide"
}

# Prompts optimized for Llama 3
PROMPT_TEMPLATES = {
    "Summary": "Provide a concise summary about {compound} and its potential for blood-brain barrier penetration. Focus on key pharmacological properties:",
    "Key Facts": "List 3-5 key facts about {compound} related to CNS drug discovery and BBB penetration:",
    "Research Papers": "Mention important research findings about {compound} and blood-brain barrier permeability:",
    "Simple Explanation": "Explain in simple terms how {compound} interacts with the blood-brain barrier:"
}
