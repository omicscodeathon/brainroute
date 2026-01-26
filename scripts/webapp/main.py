import urllib
import streamlit as st
import sys
from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Local imports
from api import get_formula, get_smiles, get_chembl_info
from utils import (load_ml_models, load_ai_model, create_chatgpt_link, generate_ai_response, 
                   create_download_link, format_batch_results_for_display, create_summary_stats)
from prediction import (predict_bbb_padel, predict_bbb_penetration_with_uncertainty, calculate_molecular_properties, 
                       process_batch_molecules)
from config import PAGE_CONFIG, PROMPT_TEMPLATES, HF_API_TOKEN
from database.quickstart import add_to_database_batch_threaded, add_to_database_threaded

# -------------------------
# Page Config & Initial Setup
# -------------------------
st.set_page_config(
    page_title="BrainRoute",
    page_icon=None,
    layout="wide"
)

# Minimal CSS - Times New Roman, black/white theme
st.markdown("""
<style>
    /* Hide sidebar and default elements */
    [data-testid="stSidebar"], [data-testid="stSidebarNav"], section[data-testid="stSidebar"] {display: none;}
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Base: Times New Roman font everywhere */
    * {
        font-family: 'Times New Roman', Times, serif !important;
    }
    
    /* Force white background on app */
    .stApp, .main, [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"] {
        background-color: #ffffff !important;
    }
    
    .main .block-container {
        max-width: 1200px;
        padding: 1rem 2rem 2rem 2rem;
    }
    
    /* All text black by default (excluding table cells - handled separately) */
    p, span, div, label, h1, h2, h3, h4, h5, h6, li, strong, em, a {
        color: #000000 !important;
    }
    
    /* Code/monospace elements - light bg, dark text */
    code, pre, .stCode, [data-testid="stCode"] {
        background-color: #f5f5f5 !important;
        color: #000000 !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 4px !important;
        padding: 2px 6px !important;
    }
    
    /* Inline code in markdown */
    .stMarkdown code {
        background-color: #f0f0f0 !important;
        color: #000000 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        font-family: 'Courier New', monospace !important;
    }
    
    /* Expanders - ensure visibility */
    [data-testid="stExpander"] {
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px !important;
        background-color: #ffffff !important;
    }
    
    [data-testid="stExpander"] summary {
        background-color: #f8f8f8 !important;
        color: #000000 !important;
        font-weight: bold !important;
        padding: 0.75rem 1rem !important;
    }
    
    [data-testid="stExpander"] summary:hover {
        background-color: #eeeeee !important;
    }
    
    [data-testid="stExpander"] summary span {
        color: #000000 !important;
    }
    
    /* Fix expander icon - hide text fallback and use CSS arrow */
    [data-testid="stExpander"] summary [data-testid="stIconMaterial"],
    [data-testid="stExpander"] summary .material-symbols-rounded {
        font-size: 0 !important;
        width: 20px !important;
        height: 20px !important;
    }
    
    [data-testid="stExpander"] summary [data-testid="stIconMaterial"]::before,
    [data-testid="stExpander"] summary .material-symbols-rounded::before {
        content: "▶" !important;
        font-size: 12px !important;
        font-family: 'Times New Roman', Times, serif !important;
        color: #000000 !important;
    }
    
    [data-testid="stExpander"][open] summary [data-testid="stIconMaterial"]::before,
    [data-testid="stExpander"][open] summary .material-symbols-rounded::before {
        content: "▼" !important;
    }
    
    [data-testid="stExpanderDetails"] {
        background-color: #ffffff !important;
        padding: 1rem !important;
    }
    
    /* Buttons - default style (nav buttons) */
    .stButton > button, .stLinkButton > a {
        background: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.2rem !important;
        color: #000000 !important;
        font-family: 'Times New Roman', Times, serif !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08) !important;
    }
    
    .stButton > button:hover, .stLinkButton > a:hover {
        background: #000000 !important;
        color: #ffffff !important;
    }
    
    .stButton > button:hover *, .stLinkButton > a:hover * {
        color: #ffffff !important;
    }
    
    /* Primary buttons */
    .stButton > button[kind="primary"], button[data-testid="baseButton-primary"] {
        background: #000000 !important;
        color: #ffffff !important;
        border: none !important;
    }
    
    .stButton > button[kind="primary"] *, button[data-testid="baseButton-primary"] * {
        color: #ffffff !important;
    }
    
    .stButton > button[kind="primary"]:hover, button[data-testid="baseButton-primary"]:hover {
        background: #333333 !important;
    }
    
    /* Secondary buttons */
    .stButton > button[kind="secondary"], button[data-testid="baseButton-secondary"] {
        background: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #000000 !important;
    }
    
    /* Form submit buttons */
    .stFormSubmitButton > button {
        background: #000000 !important;
        color: #ffffff !important;
        border: none !important;
    }
    
    .stFormSubmitButton > button * {
        color: #ffffff !important;
    }
    
    /* Header with glassmorphism */
    .main-header {
        background: rgba(230, 230, 230, 0.85);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid #cccccc;
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin: 0.5rem 0 2rem 0;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
    }
    
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: bold;
        margin: 0 0 0.75rem 0;
    }
    
    .main-header p {
        font-size: 0.95rem;
        line-height: 1.8;
        margin: 0;
    }
    
    /* Section headers */
    .section-title {
        font-size: 1.25rem;
        font-weight: bold;
        margin: 1.5rem 0 1rem 0;
    }
    
    /* Status indicators */
    .status-success {
        background: #f0f0f0;
        border-left: 3px solid #000000;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
    }
    
    /* Chat styling */
    .chat-message {
        max-width: 75%;
        padding: 0.75rem 1rem;
        border-radius: 12px;
        margin-bottom: 0.75rem;
    }
    
    .user-message {
        background: #000000 !important;
        margin-left: auto;
        margin-right: 0;
        border-bottom-right-radius: 4px;
    }
    
    .user-message, .user-message * {
        color: #ffffff !important;
        text-align: right;
    }
    
    .assistant-message {
        background: #f0f0f0 !important;
        border: 1px solid #e0e0e0;
        margin-left: 0;
        margin-right: auto;
        border-bottom-left-radius: 4px;
    }
    
    .assistant-message, .assistant-message * {
        color: #000000 !important;
        text-align: left;
    }
    
    .message-timestamp {
        font-size: 0.7rem;
        opacity: 0.7;
        margin-top: 0.25rem;
    }
    
    .chat-empty-state {
        text-align: center;
        padding: 3rem 2rem;
    }
    
    /* Text inputs */
    .stTextInput input, .stTextArea textarea {
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;
        border-radius: 8px !important;
        caret-color: #000000 !important;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #000000 !important;
        box-shadow: none !important;
        caret-color: #000000 !important;
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: #f8f8f8 !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 10px !important;
        padding: 0.75rem !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
        font-weight: bold !important;
    }
    
    /* Tables - simple styling with scroll */
    [data-testid="stTable"] {
        overflow-x: auto;
        overflow-y: auto;
        max-height: 500px;
    }
    
    [data-testid="stTable"] table {
        border-collapse: collapse;
        font-family: 'Times New Roman', Times, serif;
        white-space: nowrap;
    }
    
    [data-testid="stTable"] th,
    [data-testid="stTable"] td {
        padding: 8px 16px;
        text-align: left;
        border: 1px solid #cccccc;
        color: #000000;
        background-color: #ffffff;
        min-width: 100px;
    }
    
    [data-testid="stTable"] th {
        background-color: #f0f0f0;
        font-weight: bold;
        position: sticky;
        top: 0;
        z-index: 1;
    }
    
    /* Info/warning/error boxes */
    .stAlert, [data-testid="stAlert"] {
        background: #f8f8f8 !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 10px !important;
    }
    
    .stAlert *, [data-testid="stAlert"] * {
        color: #000000 !important;
    }
    
    /* Radio buttons */
    .stRadio label, .stRadio span {
        color: #000000 !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #f8f8f8 !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px !important;
    }
    
    [data-testid="stFileUploader"] section {
        background: #f8f8f8 !important;
    }
    
    [data-testid="stFileUploader"] section > div {
        background: #f8f8f8 !important;
    }
    
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] p {
        color: #000000 !important;
    }
    
    /* Browse files button */
    [data-testid="stFileUploader"] button {
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #e0e0e0 !important;
    }
    
    [data-testid="stFileUploader"] button * {
        color: #000000 !important;
    }
    
    [data-testid="stFileUploader"] button:hover {
        background: #000000 !important;
        color: #ffffff !important;
    }
    
    [data-testid="stFileUploader"] button:hover * {
        color: #ffffff !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #000000 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #000000 !important;
    }
    
    /* Footer */
    .app-footer {
        text-align: center;
        padding: 2rem 1rem;
        font-size: 0.8rem;
        margin-top: 2rem;
        border-top: 1px solid #e0e0e0;
    }
    
    /* Scrollable containers */
    [data-testid="stVerticalBlock"] > div {
        background-color: #ffffff !important;
    }
    
    /* Toast messages */
    [data-testid="stToast"] {
        background: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
    }
    
    [data-testid="stToast"] * {
        color: #000000 !important;
    }
    
    /* Selectbox/multiselect */
    .stSelectbox, .stMultiSelect {
        background: #ffffff !important;
    }
    
    .stSelectbox *, .stMultiSelect * {
        color: #000000 !important;
    }
    
    /* Download links */
    a.download-link {
        background: #f5f5f5 !important;
        color: #000000 !important;
        border: 1px solid #e0e0e0 !important;
        padding: 0.5rem 1rem !important;
        border-radius: 8px !important;
        text-decoration: none !important;
    }
    
    a.download-link:hover {
        background: #e0e0e0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Floating navigation buttons
nav_cols = st.columns([8, 1, 1])
with nav_cols[1]:
    st.link_button("Database", "https://omicscodeathon.github.io/brainroutedb")
with nav_cols[2]:
    if st.button("Tutorial", key="tutorial_btn"):
        st.switch_page("pages/tutorial.py")

# -------------------------
# Initialize Session State
# -------------------------
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'ai_loaded' not in st.session_state:
    st.session_state.ai_loaded = False
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'ai_model' not in st.session_state:
    st.session_state.ai_model = (None, None)
if 'current_molecule' not in st.session_state:
    st.session_state.current_molecule = None
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None
if 'processing_mode' not in st.session_state:
    st.session_state.processing_mode = "single"
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chat_input_text' not in st.session_state:
    st.session_state.chat_input_text = ""

# -------------------------
# Helper Functions
# -------------------------
def get_uncertainty_class(uncertainty):
    """Get CSS class based on uncertainty level"""
    if uncertainty < 10:
        return "uncertainty-low"
    elif uncertainty < 25:
        return "uncertainty-medium"
    else:
        return "uncertainty-high"

def get_uncertainty_interpretation(uncertainty):
    """Get human-readable interpretation of uncertainty"""
    if uncertainty < 10:
        return "Low uncertainty - High confidence in prediction"
    elif uncertainty < 25:
        return "Medium uncertainty - Moderate confidence"
    else:
        return "High uncertainty - Low confidence, consider additional validation"

def clear_prediction_results():
    """Clear prediction results when analyzing a new molecule"""
    st.session_state.prediction_results = None
    st.session_state.current_molecule = None
    st.session_state.chat_history = []

def add_to_chat(role, message):
    """Add message to chat history"""
    st.session_state.chat_history.append({
        'role': role,
        'message': message,
        'timestamp': datetime.now().strftime('%H:%M:%S')
    })

def display_chat_interface():
    """Display chat interface with messages above and input below"""
    st.markdown('<p class="section-title">Chat History</p>', unsafe_allow_html=True)
    
    with st.container(height=500):
        for chat in st.session_state.chat_history:
            if chat['role'] == 'user':
                st.markdown(f'''
                <div style="display: flex; justify-content: flex-end;">
                    <div class="chat-message user-message">
                        <div><strong>You</strong></div>
                        <div>{chat['message']}</div>
                        <div class="message-timestamp">{chat['timestamp']}</div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div style="display: flex; justify-content: flex-start;">
                    <div class="chat-message assistant-message">
                        <div><strong>Llama 3</strong></div>
                        <div>{chat['message']}</div>
                        <div class="message-timestamp">{chat['timestamp']}</div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        
        if not st.session_state.chat_history:
            st.markdown('''
                <div class="chat-empty-state">
                    <h4>Start a conversation</h4>
                    <p>Ask about your molecule's BBB properties, drug potential, or related research.</p>
                </div>
            ''', unsafe_allow_html=True)

def process_chat_question(question, compound_name, prediction, confidence):
    """Process a chat question and generate response"""
    tokenizer, model = st.session_state.ai_model
    
    context_prompt = f"""You are discussing {compound_name} with a researcher. 
    Key context: This molecule is predicted to be {prediction} for BBB penetration with {confidence:.1f}% confidence.
    Molecular properties: MW={st.session_state.prediction_results['properties']['mw']:.1f}, LogP={st.session_state.prediction_results['properties']['logp']:.2f}
    
    Question: {question}
    
    Please provide a helpful, accurate response focusing on pharmacology and drug discovery aspects."""
    
    with st.spinner("Llama 3 is thinking..."):
        response, ai_error = generate_ai_response(tokenizer, model, context_prompt)
    
    if ai_error:
        add_to_chat("assistant", f"Sorry, I encountered an error: {ai_error}")
    elif response:
        add_to_chat("assistant", response)
    else:
        add_to_chat("assistant", "I couldn't generate a response. Please try rephrasing your question.")

# -------------------------
# Model Loading Section
# -------------------------
if not st.session_state.models_loaded:
    with st.container():
        with st.spinner("Loading prediction models..."):
            models, errors = load_ml_models()
        
        if models:
            st.session_state.models = models
            st.session_state.models_loaded = True
            st.toast("Successfully loaded prediction models!")
        
        if errors:
            for error in errors:
                st.error(error)
            
            if not models:
                st.stop()

# -------------------------
# Header
# -------------------------
st.markdown('''
<div class="main-header">
    <h1 style="text-align:center;">BrainRoute</h1>
    <p style="text-align:center;">Blood-Brain Barrier Penetration Classifier<br>This tool allows you to explore molecules and predict their Blood-Brain Barrier (BBB) penetration.<br>BrainRoute will classify your molecule as BBB+ or BBB- and provide additional information for further research.</p>
</div>
''', unsafe_allow_html=True)

# -------------------------
# Processing Mode Selection
# -------------------------
if st.session_state.models_loaded:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<p class="section-title">Processing Mode</p>', unsafe_allow_html=True)
        processing_mode = st.radio(
            "Choose processing mode:",
            ["Single Molecule", "Batch Processing"],
            horizontal=True,
            key="proc_mode",
            label_visibility="collapsed"
        )
        st.session_state.processing_mode = processing_mode.lower().replace(" ", "_")

    # -------------------------
    # Single Molecule Processing
    # -------------------------
    if st.session_state.processing_mode == "single_molecule":
        st.markdown('<p class="section-title">Single Molecule Analysis</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            input_mode = st.radio(
                "Input method:",
                ["Compound Name", "SMILES String"],
                horizontal=True
            )
            st.markdown("""
            <div style="background: rgba(230, 230, 230, 0.85); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); border: 1px solid #cccccc; border-radius: 12px; padding: 1rem 1.5rem; margin: 0.5rem 0; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);">
                <p style="margin: 0 0 0.5rem 0; font-weight: bold;">Examples:</p>
                <p style="margin: 0 0 0.25rem 0;">Compound Name: aspirin, caffeine, morphine, donepezil</p>
                <p style="margin: 0;">SMILES: <code style="background: #e8e8e8; padding: 2px 6px; border-radius: 4px; font-family: monospace;">CC(=O)OC1=CC=CC=C1C(=O)O</code> (aspirin)</p>
            </div>
            """, unsafe_allow_html=True)
        
        user_input = st.text_input(
            f"Enter {input_mode}:",
            placeholder="e.g., donepezil OR CC(=O)OC1=CC=CC=C1C(=O)O",
            key="molecule_input"
        )
        
        submit_clicked = st.button("Analyze Molecule", type="primary")
        
        if user_input and user_input != st.session_state.get('last_analyzed_molecule', ''):
            clear_prediction_results()
        
        if submit_clicked or (user_input and user_input != st.session_state.get('last_input', '')):
            st.session_state.last_input = user_input
            
            if not user_input.strip():
                st.warning("Please enter a compound name or SMILES string.")
            else:
                clear_prediction_results()
                st.session_state.last_analyzed_molecule = user_input
                
                with st.spinner("Processing molecule..."):
                    mol = None
                    smiles = None
                    processing_error = None
                    
                    try:
                        if input_mode == "SMILES String":
                            mol = Chem.MolFromSmiles(user_input)
                            smiles = user_input if mol else None
                            if not mol:
                                processing_error = "Invalid SMILES string"
                        
                        elif input_mode == "Compound Name":
                            smiles = get_smiles(user_input)
                            mol = Chem.MolFromSmiles(smiles) if smiles else None
                            if not mol:
                                processing_error = f"Could not find SMILES for '{user_input}'"
                    
                    except Exception as e:
                        processing_error = f"Error processing input: {str(e)}"
                    
                    if processing_error:
                        st.error(processing_error)
                        chatgpt_link = create_chatgpt_link(user_input)
                        st.markdown(f"Try asking [ChatGPT about {user_input}]({chatgpt_link}) for more information.")
                    
                    elif mol:
                        st.session_state.current_molecule = {
                            'mol': mol,
                            'smiles': smiles,
                            'name': user_input
                        }
                        
                        with st.spinner("Fetching compound information..."):
                            info = get_chembl_info(user_input)
                            formula = get_formula(smiles)
                        
                        with st.spinner("Making BBB prediction with PaDEL-based models..."):
                            padel_preds, padel_confs, ensemble_pred, avg_conf, padel_error = predict_bbb_padel(
                                smiles, st.session_state.models
                            )
                        
                        if padel_error:
                            st.error(f"PaDEL model prediction error: {padel_error}")
                        else:
                            properties = calculate_molecular_properties(mol)
                            if properties:
                                st.session_state.prediction_results = {
                                    'padel_preds': padel_preds,
                                    'padel_confs': padel_confs,
                                    'properties': properties,
                                    'info': info,
                                    'mol': mol,
                                    'smiles': smiles,
                                    'formula': formula,
                                    'name': user_input,
                                    'prediction': ensemble_pred,
                                    'confidence': avg_conf
                                }

        # Display results if available
        if st.session_state.prediction_results:
            results = st.session_state.prediction_results
            try: 
                add_to_database_threaded(results)
            except Exception as e:
                st.toast(f"Failed to add compound to database: {e}")
            
            st.markdown('<div class="status-success">Analysis complete</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 1.5])
            
            with col1:
                st.markdown('<p class="section-title">Structure</p>', unsafe_allow_html=True)
                try:
                    mol_img = Draw.MolToImage(results['mol'], size=(300, 300))
                    st.image(mol_img, caption="Molecule Structure")
                except:
                    st.error("Could not generate structure image")
                
                if results['info']:
                    st.markdown('<p class="section-title">Compound Information</p>', unsafe_allow_html=True)
                    for key, value in results['info'].items():
                        if value and value != "Not available":
                            st.markdown(f"**{key}:** {value}")
            
            with col2:
                st.markdown('<p class="section-title">PaDEL Model Predictions</p>', unsafe_allow_html=True)
                
                pred_col1, pred_col2 = st.columns(2)
                with pred_col1:
                    st.metric("Prediction", results['prediction'])
                with pred_col2:
                    st.metric("Confidence", f"{results['confidence']:.1f}%")
                
                padel_preds = results['padel_preds']
                padel_confs = results['padel_confs']
                model_table = []
                for model in ['KNN', 'LGBM', 'ET']:
                    pred = padel_preds.get(model, None)
                    conf = padel_confs.get(model, None)
                    pred_label = 'BBB+' if pred == 1 else 'BBB-' if pred == 0 else 'N/A'
                    model_table.append({
                        'Model': model,
                        'Prediction': pred_label,
                        'Confidence (%)': f"{conf:.1f}" if conf is not None else 'N/A'
                    })
                st.table(pd.DataFrame(model_table))
                
                st.markdown('<p class="section-title">Molecular Properties</p>', unsafe_allow_html=True)
                prop_col1, prop_col2 = st.columns(2)
                with prop_col1:
                    st.metric("Molecular Weight", f"{results['properties']['mw']:.1f}")
                    st.metric("H-Bond Donors", results['properties']['hbd'])
                    st.metric("TPSA", f"{results['properties']['tpsa']:.1f}")
                with prop_col2:
                    st.metric("LogP", f"{results['properties']['logp']:.2f}")
                    st.metric("H-Bond Acceptors", results['properties']['hba'])
                    st.metric("Rotatable Bonds", results['properties']['rotatable_bonds'])
                
                st.markdown(f"**SMILES:** `{results['smiles']}`")
            
            with st.expander("Export Results"):
                preds = list(results['padel_preds'].values())
                ensemble_pred = 'BBB+' if preds.count(1) >= preds.count(0) else 'BBB-' if preds else 'N/A'
                
                result_data = {
                    'Name': results['name'],
                    'SMILES': results['smiles'],
                    'KNN Prediction': 'BBB+' if results['padel_preds'].get('KNN') == 1 else 'BBB-',
                    'KNN Confidence (%)': results['padel_confs'].get('KNN'),
                    'LGBM Prediction': 'BBB+' if results['padel_preds'].get('LGBM') == 1 else 'BBB-',
                    'LGBM Confidence (%)': results['padel_confs'].get('LGBM'),
                    'ET Prediction': 'BBB+' if results['padel_preds'].get('ET') == 1 else 'BBB-',
                    'ET Confidence (%)': results['padel_confs'].get('ET'),
                    'Ensemble Prediction': ensemble_pred,
                    'Molecular Weight': results['properties']['mw'],
                    'LogP': results['properties']['logp'],
                    'HBD': results['properties']['hbd'],
                    'HBA': results['properties']['hba'],
                    'TPSA': results['properties']['tpsa'],
                    'Rotatable Bonds': results['properties']['rotatable_bonds'],
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                export_df = pd.DataFrame([result_data])
                col_exp1, col_exp2, col_exp3 = st.columns(3)
                with col_exp1:
                    st.markdown(create_download_link(export_df, f"bbb_prediction_{results['name']}", "csv"), unsafe_allow_html=True)
                with col_exp2:
                    st.markdown(create_download_link(export_df, f"bbb_prediction_{results['name']}", "excel"), unsafe_allow_html=True)
                with col_exp3:
                    st.markdown(create_download_link(export_df, f"bbb_prediction_{results['name']}", "json"), unsafe_allow_html=True)

    # -------------------------
    # Batch Processing
    # -------------------------
    elif st.session_state.processing_mode == "batch_processing":
        st.markdown('<p class="section-title">Batch Molecule Processing</p>', unsafe_allow_html=True)
        
        input_method = st.radio(
            "Input method for batch processing:",
            ["Upload CSV File", "Text Input (Multiple Lines)"],
            horizontal=True
        )
        
        batch_input = None
        input_type = None
        
        if input_method == "Upload CSV File":
            st.info("**CSV Format:** Include columns named 'smiles' and/or 'name'")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    batch_input = pd.read_csv(uploaded_file)
                    input_type = "csv"
                    st.success(f"Loaded {len(batch_input)} molecules from CSV")
                    with st.expander("Preview uploaded data"):
                        st.table(batch_input.head())
                except Exception as e:
                    st.error(f"Error reading CSV: {str(e)}")
        
        else:
            st.info("**Format:** Enter one molecule per line (SMILES or compound names)")
            text_input = st.text_area(
                "Enter molecules (one per line):",
                placeholder="aspirin\ncaffeine\nCC(=O)OC1=CC=CC=C1C(=O)O\nmorphine",
                height=150
            )
            
            if text_input and text_input.strip():
                batch_input = text_input.strip()
                input_type = "text"
                lines = [line.strip() for line in text_input.strip().split('\n') if line.strip()]
                st.success(f"{len(lines)} molecules ready for processing")
        
        if batch_input is not None:
            if st.button("Process Batch", type="primary"):
                with st.spinner("Processing batch molecules..."):
                    
                    if input_type == "csv":
                        total_molecules = len(batch_input)
                    else:
                        total_molecules = len([line.strip() for line in batch_input.strip().split('\n') if line.strip()])
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results, batch_error = process_batch_molecules(batch_input, input_type, st.session_state.models)
                    
                    progress_bar.progress(100)
                    status_text.empty()
                    
                    if batch_error:
                        st.error(f"Batch processing error: {batch_error}")
                    else:
                        st.session_state.batch_results = results
                        st.success(f"Batch processing complete. Processed {len(results)} molecules.")
        
        if st.session_state.batch_results:
            try: 
                add_to_database_batch_threaded(st.session_state.batch_results)
            except Exception as e:
                st.toast(f"Failed to add compounds to database: {e}")
            
            st.markdown("---")
            st.markdown('<p class="section-title">Batch Results</p>', unsafe_allow_html=True)
            
            stats = create_summary_stats(st.session_state.batch_results)
            
            with st.container():
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Molecules", stats['total_molecules'])
                    st.metric("Success Rate", f"{stats['success_rate']:.1f}%")
                
                with col2:
                    st.metric("BBB+ Predictions", stats['bbb_positive_count'])
                    st.metric("BBB- Predictions", stats['bbb_negative_count'])
                
                with col3:
                    st.metric("Avg Confidence", f"{stats['avg_confidence']:.1f}%")
                    st.metric("Avg Uncertainty", f"{stats['avg_uncertainty']:.1f}%")
                
                with col4:
                    st.metric("Avg Agreement", f"{stats['avg_agreement']:.1f}%")
                    st.metric("BBB+ Rate", f"{stats['bbb_positive_rate']:.1f}%")
            
            successful_results = [r for r in st.session_state.batch_results if r.get('status') == 'Success']
            
            if successful_results:
                col1, col2 = st.columns(2)
                
                with col1:
                    pred_counts = {}
                    for result in successful_results:
                        pred = result.get('prediction', 'Unknown')
                        pred_counts[pred] = pred_counts.get(pred, 0) + 1
                    
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=list(pred_counts.keys()),
                        values=list(pred_counts.values()),
                        marker=dict(colors=['#333333', '#888888'], line=dict(color='#000000', width=2)),
                        textinfo='percent+label',
                        textfont=dict(size=14, color='#000000', family='Times New Roman')
                    )])
                    fig_pie.update_layout(
                        title=dict(text='Prediction Distribution', font=dict(color='#000000', size=16, family='Times New Roman')),
                        paper_bgcolor='#ffffff',
                        plot_bgcolor='#ffffff',
                        font=dict(family="Times New Roman", color="#000000"),
                        legend=dict(font=dict(color='#000000'))
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    conf_data = []
                    for result in successful_results:
                        conf_data.append({
                            'Confidence': result.get('confidence', 0),
                            'Uncertainty': result.get('uncertainty', 0),
                            'Prediction': result.get('prediction', 'Unknown'),
                            'Name': result.get('name', 'Unknown')
                        })
                    
                    conf_df = pd.DataFrame(conf_data)
                    
                    fig_scatter = go.Figure()
                    colors = {'BBB+': '#333333', 'BBB-': '#888888', 'Unknown': '#cccccc'}
                    for pred_type in conf_df['Prediction'].unique():
                        df_subset = conf_df[conf_df['Prediction'] == pred_type]
                        fig_scatter.add_trace(go.Scatter(
                            x=df_subset['Confidence'],
                            y=df_subset['Uncertainty'],
                            mode='markers',
                            name=pred_type,
                            marker=dict(size=10, color=colors.get(pred_type, '#666666'), line=dict(width=1, color='#000000')),
                            text=df_subset['Name'],
                            hovertemplate='%{text}<br>Confidence: %{x:.1f}%<br>Uncertainty: %{y:.1f}%<extra></extra>'
                        ))
                    
                    fig_scatter.update_layout(
                        title=dict(text='Confidence vs Uncertainty', font=dict(color='#000000', size=16, family='Times New Roman')),
                        paper_bgcolor='#ffffff',
                        plot_bgcolor='#ffffff',
                        font=dict(family="Times New Roman", color="#000000"),
                        xaxis=dict(title=dict(text='Confidence (%)', font=dict(color='#000000', size=14, family='Times New Roman')), showgrid=True, gridcolor='#cccccc', linecolor='#000000', linewidth=1, showline=True, tickfont=dict(color='#000000')),
                        yaxis=dict(title=dict(text='Uncertainty (%)', font=dict(color='#000000', size=14, family='Times New Roman')), showgrid=True, gridcolor='#cccccc', linecolor='#000000', linewidth=1, showline=True, tickfont=dict(color='#000000')),
                        legend=dict(font=dict(color='#000000'))
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
            
            display_df = format_batch_results_for_display(st.session_state.batch_results)
            
            col1, col2 = st.columns(2)
            with col1:
                status_filter = st.multiselect(
                    "Filter by Status:",
                    options=['Success', 'Error'],
                    default=['Success', 'Error']
                )
            with col2:
                if 'Prediction' in display_df.columns:
                    pred_filter = st.multiselect(
                        "Filter by Prediction:",
                        options=['BBB+', 'BBB-'],
                        default=['BBB+', 'BBB-']
                    )
                else:
                    pred_filter = []
            
            filtered_df = display_df[display_df['Status'].isin(status_filter)]
            if pred_filter and 'Prediction' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Prediction'].isin(pred_filter)]
            
            st.table(filtered_df)
            
            
            st.markdown('<p class="section-title">Export Batch Results</p>', unsafe_allow_html=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(create_download_link(filtered_df, f"bbb_batch_results_{timestamp}", "csv"), unsafe_allow_html=True)
            with col2:
                st.markdown(create_download_link(filtered_df, f"bbb_batch_results_{timestamp}", "excel"), unsafe_allow_html=True)
            with col3:
                st.markdown(create_download_link(filtered_df, f"bbb_batch_results_{timestamp}", "json"), unsafe_allow_html=True)

# -------------------------
# AI Chat Section
# -------------------------
if st.session_state.prediction_results:
    st.markdown("---")
    st.markdown('<p class="section-title">Chat with Llama 3 about Your Molecule</p>', unsafe_allow_html=True)
    
    if not st.session_state.ai_loaded:
        if not HF_API_TOKEN:
            st.error("Hugging Face API token not configured")
            st.info("""
            **Setup Instructions:**
            1. Get a free token from [Hugging Face](https://huggingface.co/settings/tokens)
            2. Set environment variable: `HUGGINGFACE_API_TOKEN=your_token_here`
            3. Restart the application
            """)
        else:
            col1, col2 = st.columns([2, 1])
            with col1:
                if st.button("Connect to Llama 3", type="secondary"):
                    with st.spinner("Connecting to Llama 3 via Hugging Face API..."):
                        tokenizer, model, ai_error = load_ai_model()
                        
                    if ai_error:
                        st.error(f"Failed to connect to AI model: {ai_error}")
                    else:
                        st.session_state.ai_model = (tokenizer, model)
                        st.session_state.ai_loaded = True
                        st.success("Connected to Llama 3 API")
                        st.rerun()
            
            with col2:
                st.info("Chat with AI about your molecule's BBB properties, drug potential, and more.")
    
    if st.session_state.ai_loaded:
        tokenizer, model = st.session_state.ai_model
        
        if tokenizer and model:
            compound_name = st.session_state.prediction_results['name']
            comp = st.session_state.prediction_results['info']['ChEMBL ID']
            preds = list(st.session_state.prediction_results['padel_preds'].values())
            prediction = 'BBB+' if preds.count(1) >= preds.count(0) else 'BBB-' if preds else 'N/A'
            confs = [c for c in st.session_state.prediction_results['padel_confs'].values() if c is not None]
            confidence = sum(confs) / len(confs) if confs else 0.0
            
            st.success(f"**Llama 3 is ready.** Ask anything about **{compound_name}** (Predicted: {prediction}, Confidence: {confidence:.1f}%)")
            
            display_chat_interface()
            
            st.markdown("**Quick Questions:**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("Drug Potential", key="drug_potential"):
                    question = f"What is the drug potential of {compound_name} for CNS disorders? Consider its BBB penetration."
                    add_to_chat("user", question)
                    st.rerun()
        
            with col2:
                if st.button("Properties", key="properties"):
                    question = f"Explain the key molecular properties of {compound_name} that affect its BBB penetration."
                    add_to_chat("user", question)
                    st.rerun()
        
            with col3:
                if st.button("Side Effects", key="side_effects"):
                    question = f"What are the potential side effects and safety concerns for {compound_name}?"
                    add_to_chat("user", question)
                    st.rerun()
        
            with col4:
                if st.button("Research", key="research"):
                    question = f"What current research exists on {compound_name} for brain-related diseases?"
                    add_to_chat("user", question)
                    st.rerun()
            
            st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
            
            with st.form(key="chat_form", clear_on_submit=True):
                col_input, col_send, col_clear = st.columns([6, 1, 1])
                
                with col_input:
                    chat_question = st.text_input(
                        "Your message:",
                        placeholder=f"e.g., How does {compound_name} work in the brain?",
                        label_visibility="collapsed",
                        key="chat_input_field"
                    )
                
                with col_send:
                    send_clicked = st.form_submit_button("Send", type="primary", use_container_width=True)
                
                with col_clear:
                    clear_clicked = st.form_submit_button("Clear", use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if clear_clicked:
                st.session_state.chat_history = []
                st.rerun()
            
            if send_clicked and chat_question.strip():
                add_to_chat("user", chat_question)
                st.rerun()
            
            if st.session_state.chat_history and st.session_state.chat_history[-1]['role'] == 'user':
                last_question = st.session_state.chat_history[-1]['message']
                process_chat_question(last_question, compound_name, prediction, confidence)
                st.rerun()
            
            if st.session_state.chat_history:
                with st.expander("Export Chat History"):
                    chat_df = pd.DataFrame([
                        {
                            'Timestamp': chat['timestamp'],
                            'Role': chat['role'].title(),
                            'Message': chat['message'],
                            'Molecule': compound_name
                        }
                        for chat in st.session_state.chat_history
                    ])
                    st.markdown(create_download_link(chat_df, f"chat_history_{compound_name}", "csv"), unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown('<p class="section-title">Additional Resources</p>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                chatgpt_link = create_chatgpt_link(compound_name)
                st.markdown(f"[ChatGPT Analysis]({chatgpt_link})")
            
            with col2:
                pubmed_link = f"https://pubmed.ncbi.nlm.nih.gov/?term={urllib.parse.quote(compound_name + ' blood brain barrier')}"
                st.markdown(f"[PubMed Search]({pubmed_link})")
            
            with col3:
                chembl_link = f"https://www.ebi.ac.uk/chembl/compound_report_card/{comp}/"
                st.markdown(f"[ChEMBL Database]({chembl_link})")

# -------------------------
# Footer
# -------------------------
st.markdown("""
<div class="app-footer">
    BrainRoute v2025.01 | Omics-Codeathon
</div>
""", unsafe_allow_html=True)

print(f"Python executable: {sys.executable}")
