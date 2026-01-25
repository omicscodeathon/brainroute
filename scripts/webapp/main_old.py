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
st.set_page_config(**PAGE_CONFIG)

# Custom CSS for better UI with improved chat interface
st.markdown("""
<style>
    .main-header {
        text-align: center
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .uncertainty-high { color: #e74c3c; font-weight: bold; }
    .uncertainty-medium { color: #f39c12; font-weight: bold; }
    .uncertainty-low { color: #27ae60; font-weight: bold; }
    .status-success {
        padding: 1rem;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .status-error {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .batch-stats {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #007bff;
    }

    /* Target the Streamlit container to make it scrollable from the bottom */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        overflow-y: auto;
        display: flex;
        flex-direction: column;
    }
    
    /* Enhanced Chat Interface Styles */
    .chat-container-inner {
        display: flex;
        flex-direction: column; /* Match the container's direction */
        gap: 0rem;
    }
    
    .chat-message {
        padding: 0.4rem 0.9rem; /* Reduced vertical padding for a shorter bubble */
        border-radius: 18px;
        width: fit-content; /* CRITICAL: Make bubble width dynamic */
        max-width: 75%; /* Prevent bubbles from being too wide on large screens */
        word-wrap: break-word;
        animation: slideIn 0.3s ease-out;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        margin-bottom: 0.75rem; /* Add margin to the bottom of each message */
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .user-message {
        background: linear-gradient(135deg, #005c4b 0%, #008069 100%); /* WhatsApp green gradient */
        color: white;
        align-self: flex-end; /* Align to the right */
        border-bottom-right-radius: 6px;
        margin-left: auto; /* Push to the right */
        margin-right: 0;
    }
    
    .assistant-message {
        background-color: white;
        color: #2c3e50;
        align-self: flex-start; /* Align to the left */
        border: 1px solid #e1e8ed;
        border-bottom-left-radius: 6px;
        margin-right: auto; /* Push to the left */
        margin-left: 0;
    }
    
    .message-timestamp {
        font-size: 0.7rem;
        opacity: 0.7;
        margin-top: 0.25rem;
    }
    
    .chat-empty-state {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
    }
    
    .chat-input-container {
        background-color: white;
        border: 2px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    .quick-question-btn {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 20px;
        cursor: pointer;
        transition: all 0.2s;
        font-size: 0.9rem;
    }
    
    .quick-question-btn:hover {
        background-color: #e9ecef;
        border-color: #adb5bd;
    }
</style>
<script>
    // JavaScript to scroll the chat container to the bottom on new messages
    function scrollToBottom() {
        // Find the specific scrollable container Streamlit creates
        const chatContainer = window.parent.document.querySelector('[data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"]');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }

    // This is a bit of a hack to run the script after Streamlit has updated the DOM.
    // We use a component that does nothing but trigger the script on re-render.
    const streamlitDoc = window.parent.document;
    if (!streamlitDoc.querySelector('.scroll-observer')) {
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.addedNodes.length) {
                    scrollToBottom();
                }
            });
        });
        observer.observe(streamlitDoc.body, { childList: true, subtree: true });
        
        // Add a marker to prevent re-adding the observer
        const marker = streamlitDoc.createElement('div');
        marker.className = 'scroll-observer';
        marker.style.display = 'none';
        streamlitDoc.body.appendChild(marker);
    }
    
    // Also run on initial load
    window.addEventListener('load', scrollToBottom);
</script>
""", unsafe_allow_html=True)

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
    st.markdown("### 💬 Chat History")
    
    # Use Streamlit's native container for robust scrolling
    with st.container(height=600):
        # The inner div helps with styling and message order
        st.markdown('<div class="chat-container-inner">', unsafe_allow_html=True)
        
        # Display messages in standard chronological order (oldest first)
        for chat in st.session_state.chat_history:
            if chat['role'] == 'user':
                st.markdown(f'''
                <div class="chat-message user-message">
                    <div><strong>You</strong></div>
                    <div>{chat['message']}</div>
                    <div class="message-timestamp">{chat['timestamp']}</div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="chat-message assistant-message">
                    <div><strong>🦙 Llama 3</strong></div>
                    <div>{chat['message']}</div>
                    <div class="message-timestamp">{chat['timestamp']}</div>
                </div>
                ''', unsafe_allow_html=True)
        
        if not st.session_state.chat_history:
            st.markdown('''
                <div class="chat-empty-state">
                    <h4>👋 Start a conversation!</h4>
                    <p>Ask me anything about your molecule's BBB properties, drug potential, or related research.</p>
                </div>
            ''', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

def process_chat_question(question, compound_name, prediction, confidence):
    """Process a chat question and generate response"""
    # This function now only generates the AI response
    tokenizer, model = st.session_state.ai_model
    
    # Create context-aware prompt
    context_prompt = f"""You are discussing {compound_name} with a researcher. 
    Key context: This molecule is predicted to be {prediction} for BBB penetration with {confidence:.1f}% confidence.
    Molecular properties: MW={st.session_state.prediction_results['properties']['mw']:.1f}, LogP={st.session_state.prediction_results['properties']['logp']:.2f}
    
    Question: {question}
    
    Please provide a helpful, accurate response focusing on pharmacology and drug discovery aspects."""
    
    with st.spinner("🦙 Llama 3 is thinking..."):
        response, ai_error = generate_ai_response(tokenizer, model, context_prompt)
    
    if ai_error:
        add_to_chat("assistant", f"❌ Sorry, I encountered an error: {ai_error}")
    elif response:
        add_to_chat("assistant", response)
    else:
        add_to_chat("assistant", "❓ I couldn't generate a response. Please try rephrasing your question.")

# -------------------------
# Model Loading Section
# -------------------------
if not st.session_state.models_loaded:
       
    with st.container():
        with st.spinner("🔄 Loading Prediction Models"):
            models, errors = load_ml_models()
        
        if models:
            st.session_state.models = models
            st.session_state.models_loaded = True
            st.success(f"✅ Successfully loaded {len(models)} models: {', '.join(models.keys())}")
        
        if errors:
            for error in errors:
                st.error(f"❌ {error}")
            
            if not models:
                st.stop()
# -------------------------
# Header
# -------------------------
st.markdown('<div class="main-header"><h1 style="text-align:center;">🧠 BrainRoute</h1><p style="text-align:center;">Blood-Brain Barrier Penetration Classifier<br>This tool allows you to explore molecules and predict their Blood-Brain Barrier (BBB) penetration.<br>BrainRoute will classify your molecule as BBB+ or BBB- and provide additional information about the molecule for further drug discovery research!</p></div>', unsafe_allow_html=True)


# -------------------------
# Processing Mode Selection
# -------------------------
if st.session_state.models_loaded:
    mode_col, db_col = st.columns(2)
    with mode_col:
        st.subheader("🎯 Processing Mode")
        processing_mode = st.radio(
            "Choose processing mode:",
            ["Single Molecule", "Batch Processing"],
            horizontal=True,
            key="proc_mode",
            label_visibility="collapsed"
        )
        
        st.session_state.processing_mode = processing_mode.lower().replace(" ", "_")
    with db_col:
        st.info('Access Database')
        st.link_button("🔗 Go to Database Site", "https://omicscodeathon.github.io/brainroutedb", use_container_width=True)

    # -------------------------
    # Single Molecule Processing
    # -------------------------
    if st.session_state.processing_mode == "single_molecule":
        st.subheader("🔬 Single Molecule Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            input_mode = st.radio(
                "Input method:",
                ["Compound Name", "SMILES String"],
                horizontal=True
            )
            st.info("""
            **Examples:**
            - **Compound Name:** aspirin, caffeine, morphine, donepezil
            - **SMILES:** CC(=O)OC1=CC=CC=C1C(=O)O (aspirin)
            """)
        
        user_input = st.text_input(
            f"Enter {input_mode}:",
            placeholder="e.g., donepezil OR CC(=O)OC1=CC=CC=C1C(=O)O",
            key="molecule_input"
        )
        
        submit_clicked = st.button("🚀 Analyze Molecule", type="primary")
        
        # Check if this is a new molecule (clear results if so)
        if user_input and user_input != st.session_state.get('last_analyzed_molecule', ''):
            clear_prediction_results()
        
        if submit_clicked or (user_input and user_input != st.session_state.get('last_input', '')):
            st.session_state.last_input = user_input
            
            if not user_input.strip():
                st.warning("⚠️ Please enter a compound name or SMILES string.")
            else:
                # Clear previous results for new molecule
                clear_prediction_results()
                st.session_state.last_analyzed_molecule = user_input
                
                with st.spinner("🔍 Processing molecule..."):
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
                        st.error(f"❌ {processing_error}")
                        chatgpt_link = create_chatgpt_link(user_input)
                        st.markdown(f"💡 Try asking [ChatGPT about {user_input}]({chatgpt_link}) for more information.")
                    
                    elif mol:
                        st.session_state.current_molecule = {
                            'mol': mol,
                            'smiles': smiles,
                            'name': user_input
                        }
                        
                        # Get ChEMBL info
                        with st.spinner("📚 Fetching compound information..."):
                            info = get_chembl_info(user_input)
                            formula = get_formula(smiles)
                        
                        # Make prediction with uncertainty
                        with st.spinner("🤖 Making BBB prediction with PaDEL-based models..."):
                            padel_preds, padel_confs, ensemble_pred, avg_conf, padel_error = predict_bbb_padel(
                                smiles, st.session_state.models
                            )
                        if padel_error:
                            st.error(f"❌ PaDEL model prediction error: {padel_error}")
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

        # Display results if available (persistent across interactions)
        if st.session_state.prediction_results:
            results = st.session_state.prediction_results
            try: 
                add_to_database_threaded(results)
            except Exception as e:
                st.toast(f"Failed to add compound to database:{e}", icon="⚠️")
            st.success("✅ Analysis complete!")
            
            # Results layout
            col1, col2 = st.columns([1, 1.5])
            
            with col1:
                st.subheader("🧬 Structure")
                try:
                    mol_img = Draw.MolToImage(results['mol'], size=(300, 300))
                    st.image(mol_img, caption="Molecule Structure")
                except:
                    st.error("Could not generate structure image")
                
                # ChEMBL info if available
                if results['info']:
                    st.markdown("#### 📋 Compound Information")
                    for key, value in results['info'].items():
                        if value and value != "Not available":
                            st.markdown(f"**{key}:** {value}")
            
            with col2:
                st.subheader("📊 PaDEL Model Predictions")
                
                pred_col1, pred_col2 = st.columns(2)
                with pred_col1:
                    st.metric("Prediction", results['prediction'])
                with pred_col2:
                    st.metric("Confidence", f"{results['confidence']:.1f}")
                
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
                st.dataframe(pd.DataFrame(model_table))
                
                # Molecular properties
                st.markdown("#### 🧪 Molecular Properties")
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
            
            # Export single result
            with st.expander("📥 Export Results"):
                # Compute ensemble/majority prediction
                preds = list(results['padel_preds'].values())
                if preds:
                    ensemble_pred = 'BBB+' if preds.count(1) >= preds.count(0) else 'BBB-'
                else:
                    ensemble_pred = 'N/A'
                # Prepare export data with all model predictions/confidences
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
        st.subheader("📋 Batch Molecule Processing")
        
        input_method = st.radio(
            "Input method for batch processing:",
            ["Upload CSV File", "Text Input (Multiple Lines)"],
            horizontal=True
        )
        
        batch_input = None
        input_type = None
        
        if input_method == "Upload CSV File":
            st.info("📁 **CSV Format:** Include columns named 'smiles' and/or 'name'")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    batch_input = pd.read_csv(uploaded_file)
                    input_type = "csv"
                    st.success(f"✅ Loaded {len(batch_input)} molecules from CSV")
                    with st.expander("Preview uploaded data"):
                        st.dataframe(batch_input.head())
                except Exception as e:
                    st.error(f"❌ Error reading CSV: {str(e)}")
        
        else:  # Text Input
            st.info("📝 **Format:** Enter one molecule per line (SMILES or compound names)")
            text_input = st.text_area(
                "Enter molecules (one per line):",
                placeholder="aspirin\ncaffeine\nCC(=O)OC1=CC=CC=C1C(=O)O\nmorphine",
                height=150
            )
            
            if text_input and text_input.strip():
                batch_input = text_input.strip()
                input_type = "text"
                lines = [line.strip() for line in text_input.strip().split('\n') if line.strip()]
                st.success(f"✅ {len(lines)} molecules ready for processing")
        # Process batch
        if batch_input is not None:
            if st.button("🚀 Process Batch", type="primary"):
                with st.spinner("🔄 Processing batch molecules..."):
                    
                    # Create progress bar
                    if input_type == "csv":
                        total_molecules = len(batch_input)
                    else:
                        total_molecules = len([line.strip() for line in batch_input.strip().split('\n') if line.strip()])
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process molecules
                    results, batch_error = process_batch_molecules(batch_input, input_type, st.session_state.models)
                    
                    progress_bar.progress(100)
                    status_text.empty()
                    
                    if batch_error:
                        st.error(f"❌ Batch processing error: {batch_error}")
                    else:
                        st.session_state.batch_results = results
                        st.success(f"✅ Batch processing complete! Processed {len(results)} molecules.")
        
        # Display batch results
        if st.session_state.batch_results:
            try: 
                add_to_database_batch_threaded(results)
            except Exception as e:
                st.toast(f"Failed to add compounds to database:{e}", icon="⚠️")
            st.success("✅ Analysis complete!")
            st.markdown("---")
            st.subheader("📊 Batch Results")
            
            # Summary statistics
            stats = create_summary_stats(st.session_state.batch_results)
            
            with st.container():
                st.markdown('<div class="batch-stats">', unsafe_allow_html=True)
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
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Results visualization
            successful_results = [r for r in st.session_state.batch_results if r.get('status') == 'Success']
            
            if successful_results:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Prediction distribution pie chart
                    pred_counts = {}
                    for result in successful_results:
                        pred = result.get('prediction', 'Unknown')
                        pred_counts[pred] = pred_counts.get(pred, 0) + 1
                    
                    fig_pie = px.pie(
                        values=list(pred_counts.values()),
                        names=list(pred_counts.keys()),
                        title="Prediction Distribution",
                        color_discrete_map={"BBB+": "#27ae60", "BBB-": "#e74c3c"}
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Confidence vs Uncertainty scatter plot
                    conf_data = []
                    for result in successful_results:
                        conf_data.append({
                            'Confidence': result.get('confidence', 0),
                            'Uncertainty': result.get('uncertainty', 0),
                            'Prediction': result.get('prediction', 'Unknown'),
                            'Name': result.get('name', 'Unknown')
                        })
                    
                    conf_df = pd.DataFrame(conf_data)
                    fig_scatter = px.scatter(
                        conf_df, x='Confidence', y='Uncertainty',
                        color='Prediction', title="Confidence vs Uncertainty",
                        hover_data=['Name'],
                        color_discrete_map={"BBB+": "#27ae60", "BBB-": "#e74c3c"}
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Results table
            display_df = format_batch_results_for_display(st.session_state.batch_results)
            
            # Filter options
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
            
            # Apply filters
            filtered_df = display_df[display_df['Status'].isin(status_filter)]
            if pred_filter and 'Prediction' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Prediction'].isin(pred_filter)]
            
            st.dataframe(filtered_df, use_container_width=True)
            
            # Export batch results
            st.markdown("### 📥 Export Batch Results")
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(create_download_link(filtered_df, f"bbb_batch_results_{timestamp}", "csv"), unsafe_allow_html=True)
            with col2:
                st.markdown(create_download_link(filtered_df, f"bbb_batch_results_{timestamp}", "excel"), unsafe_allow_html=True)
            with col3:
                st.markdown(create_download_link(filtered_df, f"bbb_batch_results_{timestamp}", "json"), unsafe_allow_html=True)
         
# -------------------------
# AI Chat Section - Enhanced Chat Interface with Input Below
# -------------------------
if st.session_state.prediction_results:
    st.markdown("---")
    st.subheader("💬 Chat with Llama 3 about Your Molecule")
    
    if not st.session_state.ai_loaded:
        # Check API setup
        if not HF_API_TOKEN:
            st.error("❌ Hugging Face API token not configured")
            st.info("""
            **Setup Instructions:**
            1. Get a free token from [Hugging Face](https://huggingface.co/settings/tokens)
            2. Set environment variable: `HUGGINGFACE_API_TOKEN=your_token_here`
            3. Restart the application
            """)
        else:
            col1, col2 = st.columns([2, 1])
            with col1:
                if st.button("🚀 Connect to Llama 3", type="secondary"):
                    with st.spinner("🔌 Connecting to Llama 3 via Hugging Face API..."):
                        tokenizer, model, ai_error = load_ai_model()
                        
                    if ai_error:
                        st.error(f"❌ Failed to connect to AI model: {ai_error}")
                    else:
                        st.session_state.ai_model = (tokenizer, model)
                        st.session_state.ai_loaded = True
                        st.success("✅ Connected to Llama 3 API!")
                        st.rerun()
            
            with col2:
                st.info("🦙 Chat with AI about your molecule's BBB properties, drug potential, and more!")
    
    if st.session_state.ai_loaded:
        tokenizer, model = st.session_state.ai_model
        
        if tokenizer and model:
            compound_name = st.session_state.prediction_results['name']
            comp = st.session_state.prediction_results['info']['ChEMBL ID']
            # Use ensemble/majority prediction for chat context
            preds = list(st.session_state.prediction_results['padel_preds'].values())
            if preds:
                prediction = 'BBB+' if preds.count(1) >= preds.count(0) else 'BBB-'
            else:
                prediction = 'N/A'
            # Use average confidence for chat context
            confs = [c for c in st.session_state.prediction_results['padel_confs'].values() if c is not None]
            confidence = sum(confs) / len(confs) if confs else 0.0
            st.success(f"🦙 **Llama 3 is ready!** Ask anything about **{compound_name}** (Predicted: {prediction}, Confidence: {confidence:.1f}%)")
            
            # Display chat history first (messages above)
            display_chat_interface()
            
            # Quick action buttons below chat
            st.markdown("**💡 Quick Questions:**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("💊 Drug Potential", key="drug_potential"):
                    st.session_state.chat_input_text = f"What is the drug potential of {compound_name} for CNS disorders? Consider its BBB penetration."
                    st.rerun()
        
            with col2:
                if st.button("🧪 Properties", key="properties"):
                    st.session_state.chat_input_text = f"Explain the key molecular properties of {compound_name} that affect its BBB penetration."
                    st.rerun()
        
            with col3:
                if st.button("⚠️ Side Effects", key="side_effects"):
                    st.session_state.chat_input_text = f"What are the potential side effects and safety concerns for {compound_name}?"
                    st.rerun()
        
            with col4:
                if st.button("🔬 Research", key="research"):
                    st.session_state.chat_input_text = f"What current research exists on {compound_name} for brain-related diseases?"
                    st.rerun()
                
            # # Process pending question if exists
            # if st.session_state.pending_question:
            #     question = st.session_state.pending_question
            #     st.session_state.pending_question = None
            #     process_chat_question(question, compound_name, prediction, confidence)
            #     st.rerun()
            
            # Chat input container at the bottom with improved styling
            st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
            
            # Use form for better UX
            with st.form(key="chat_form", clear_on_submit=True):
                col_input, col_send, col_clear = st.columns([6, 1, 1])
                
                with col_input:
                    chat_question = st.text_input(
                        "Your message:",
                        value=st.session_state.chat_input_text,
                        placeholder=f"e.g., How does {compound_name} work in the brain?",
                        label_visibility="collapsed",
                        key="chat_input_field"
                    )
                
                with col_send:
                    send_clicked = st.form_submit_button("📤 Send", type="primary", use_container_width=True)
                
                with col_clear:
                    clear_clicked = st.form_submit_button("🗑️", use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Handle clear button
            if clear_clicked:
                st.session_state.chat_history = []
                st.rerun()
            
            # Handle send button
            if send_clicked and chat_question.strip():
                # 1. Immediately add user message and rerun
                add_to_chat("user", chat_question)
                st.rerun()
            
            # 2. If the last message was from the user, generate AI response
            if st.session_state.chat_history and st.session_state.chat_history[-1]['role'] == 'user':
                last_question = st.session_state.chat_history[-1]['message']
                process_chat_question(last_question, compound_name, prediction, confidence)
                st.rerun() # Rerun again to display the new AI message
            
            # Export chat history
            if st.session_state.chat_history:
                with st.expander("📥 Export Chat History"):
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
            
            # Alternative research options
            st.markdown("---")
            st.markdown("### 🔍 Additional Resources")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                chatgpt_link = create_chatgpt_link(compound_name)
                st.markdown(f"[💬 ChatGPT Analysis]({chatgpt_link})")
            
            with col2:
                pubmed_link = f"https://pubmed.ncbi.nlm.nih.gov/?term={urllib.parse.quote(compound_name + ' blood brain barrier')}"
                st.markdown(f"[📚 PubMed Search]({pubmed_link})")
            
            with col3:
                chembl_link = f"https://www.ebi.ac.uk/chembl/compound_report_card/{comp}/"
                st.markdown(f"[🧬 ChEMBL Database]({chembl_link})")


# -------------------------
# Footer
# -------------------------
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.caption("NeuroGate v2025.01 | Omics-Codeathon | © 2025 NeuroGate Team")
print(f"Python executable: {sys.executable}")