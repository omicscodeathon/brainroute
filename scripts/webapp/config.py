import streamlit as st

# Model Configuration
MODEL_PATHS = {
    'KNN': '../../output/models/padel_KNN_model.pkl',
    'LGBM': '../../output/models/padel_LGBM_model.pkl',
    'ET': '../../output/models/padel_ET_model.pkl',
    'scaler': '../../output/models/scaler_padel.pkl',
}

# AI Model Configuration - Using OpenAI client with HF router
AI_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct:novita"
USE_HF_INFERENCE_API = True
DEFAULT_MODEL = "KNN"

# Hugging Face API Configuration
HF_API_TOKEN = st.secrets["HF_TOKEN"]

# Generation Parameters for OpenAI client
API_GENERATION_CONFIG = {
    'max_new_tokens': 150,
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