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
from api import get_smiles, get_chembl_info
from utils import (load_ml_models, load_ai_model, create_chatgpt_link, generate_ai_response, 
                   create_download_link, format_batch_results_for_display, create_summary_stats)
from prediction import (predict_bbb_penetration_with_uncertainty, calculate_molecular_properties, 
                       process_batch_molecules)
from config import PAGE_CONFIG, PROMPT_TEMPLATES, HF_API_TOKEN

# -------------------------
# Page Config & Initial Setup
# -------------------------
st.set_page_config(**PAGE_CONFIG)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
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
</style>
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

# -------------------------
# Header
# -------------------------
st.markdown('<div class="main-header"><h1>🧠 NeuroGate</h1><p>Blood-Brain Barrier Penetration Classifier<br>This tool allows you to explore molecules and predict their Blood-Brain Barrier (BBB) penetration.<br>Neurogate will classify your molecule as BBB+ or BBB- and provide additional information about the molecule for further drug discovery research!</p></div>', unsafe_allow_html=True)

# -------------------------
# Model Loading Section
# -------------------------
if not st.session_state.models_loaded:
    st.info("🔄 Initializing models for optimal performance...")
    
    with st.container():
        st.subheader("🔄 Loading Prediction Models")
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
# Processing Mode Selection
# -------------------------
if st.session_state.models_loaded:
    st.subheader("🎯 Processing Mode")
    
    processing_mode = st.radio(
        "Choose processing mode:",
        ["Single Molecule", "Batch Processing"],
        horizontal=True,
        key="proc_mode"
    )
    
    st.session_state.processing_mode = processing_mode.lower().replace(" ", "_")

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
        
        if submit_clicked or (user_input and user_input != st.session_state.get('last_input', '')):
            st.session_state.last_input = user_input
            
            if not user_input.strip():
                st.warning("⚠️ Please enter a compound name or SMILES string.")
            else:
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
                        
                        # Make prediction with uncertainty
                        with st.spinner("🤖 Making BBB prediction with uncertainty analysis..."):
                            pred_result, pred_error = predict_bbb_penetration_with_uncertainty(
                                mol, st.session_state.models
                            )
                        
                        if pred_error:
                            st.error(f"❌ Prediction error: {pred_error}")
                        else:
                            properties = calculate_molecular_properties(mol)
                            
                            if properties:
                                st.success("✅ Analysis complete!")
                                
                                # Results layout
                                col1, col2 = st.columns([1, 1.5])
                                
                                with col1:
                                    st.subheader("🧬 Structure")
                                    try:
                                        mol_img = Draw.MolToImage(mol, size=(300, 300))
                                        st.image(mol_img, caption="Molecule Structure")
                                    except:
                                        st.error("Could not generate structure image")
                                    
                                    # ChEMBL info if available
                                    if info:
                                        st.markdown("#### 📋 Compound Information")
                                        for key, value in info.items():
                                            if value and value != "Not available":
                                                st.markdown(f"**{key}:** {value}")
                                
                                with col2:
                                    st.subheader("📊 Prediction Results")
                                    
                                    # Main prediction with uncertainty
                                    pred_color = "green" if pred_result['prediction'] == "BBB+" else "red"
                                    st.markdown(f"### 🎯 Prediction: <span style='color:{pred_color}'>{pred_result['prediction']}</span>", unsafe_allow_html=True)
                                    
                                    # Confidence and uncertainty metrics
                                    col_a, col_b, col_c = st.columns(3)
                                    with col_a:
                                        st.metric("Confidence", f"{pred_result['confidence']:.1f}%")
                                    with col_b:
                                        uncertainty_class = get_uncertainty_class(pred_result['uncertainty'])
                                        st.markdown(f"**Uncertainty:** <span class='{uncertainty_class}'>{pred_result['uncertainty']:.1f}%</span>", unsafe_allow_html=True)
                                    with col_c:
                                        st.metric("Model Agreement", f"{pred_result['agreement']:.1f}%")
                                    
                                    # Uncertainty interpretation
                                    uncertainty_text = get_uncertainty_interpretation(pred_result['uncertainty'])
                                    st.info(f"🔍 **Uncertainty Analysis:** {uncertainty_text}")
                                    
                                    # Individual model results
                                    with st.expander("🔧 Individual Model Results"):
                                        individual_df = []
                                        for model_name, confidence in pred_result['individual_confidences'].items():
                                            pred_class = pred_result['individual_predictions'][model_name]
                                            individual_df.append({
                                                'Model': model_name,
                                                'Prediction': "BBB+" if pred_class == 1 else "BBB-",
                                                'BBB+ Prob': f"{confidence[1]:.3f}",
                                                'BBB- Prob': f"{confidence[0]:.3f}"
                                            })
                                        st.dataframe(pd.DataFrame(individual_df))
                                    
                                    # Molecular properties
                                    st.markdown("#### 🧪 Molecular Properties")
                                    prop_col1, prop_col2 = st.columns(2)
                                    with prop_col1:
                                        st.metric("Molecular Weight", f"{properties['mw']:.1f}")
                                        st.metric("H-Bond Donors", properties['hbd'])
                                        st.metric("TPSA", f"{properties['tpsa']:.1f}")
                                    with prop_col2:
                                        st.metric("LogP", f"{properties['logp']:.2f}")
                                        st.metric("H-Bond Acceptors", properties['hba'])
                                        st.metric("Rotatable Bonds", properties['rotatable_bonds'])
                                    
                                    st.markdown(f"**SMILES:** `{smiles}`")
                                    
                                
                                # Export single result
                                with st.expander("📥 Export Results"):
                                    result_data = {
                                        'Name': user_input,
                                        'SMILES': smiles,
                                        'Prediction': pred_result['prediction'],
                                        'Confidence (%)': pred_result['confidence'],
                                        'Uncertainty (%)': pred_result['uncertainty'],
                                        'Agreement (%)': pred_result['agreement'],
                                        'Molecular Weight': properties['mw'],
                                        'LogP': properties['logp'],
                                        'HBD': properties['hbd'],
                                        'HBA': properties['hba'],
                                        'TPSA': properties['tpsa'],
                                        'Rotatable Bonds': properties['rotatable_bonds'],
                                        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    }
                                    
                                    export_df = pd.DataFrame([result_data])
                                    
                                    col_exp1, col_exp2, col_exp3 = st.columns(3)
                                    with col_exp1:
                                        st.markdown(create_download_link(export_df, f"bbb_prediction_{user_input}", "csv"), unsafe_allow_html=True)
                                    with col_exp2:
                                        st.markdown(create_download_link(export_df, f"bbb_prediction_{user_input}", "excel"), unsafe_allow_html=True)
                                    with col_exp3:
                                        st.markdown(create_download_link(export_df, f"bbb_prediction_{user_input}", "json"), unsafe_allow_html=True)

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
            
            if text_input.strip():
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
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(create_download_link(filtered_df, f"bbb_batch_results_{timestamp}", "csv"), unsafe_allow_html=True)
            with col2:
                st.markdown(create_download_link(filtered_df, f"bbb_batch_results_{timestamp}", "excel"), unsafe_allow_html=True)
            with col3:
                st.markdown(create_download_link(filtered_df, f"bbb_batch_results_{timestamp}", "json"), unsafe_allow_html=True)
            with col4:
                # Full detailed results
                full_results_df = pd.DataFrame(st.session_state.batch_results)
                st.markdown(create_download_link(full_results_df, f"bbb_detailed_results_{timestamp}", "csv"), unsafe_allow_html=True)


# -------------------------
# AI Analysis Section - Updated for HF API
# -------------------------
if st.session_state.current_molecule:
    st.markdown("---")
    st.subheader("🤖 AI-Powered Analysis with Llama 3")
    
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
    
    if st.session_state.ai_loaded:
        tokenizer, model = st.session_state.ai_model
        
        if tokenizer and model:
            st.info("🦙 **Powered by Llama 3 8B Instruct** - Advanced AI analysis for molecular insights")
            
            st.markdown("**Choose response style:**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            response_generated = False
            
            with col1:
                if st.button("📝 Summary", key="summary"):
                    response_type = "Summary"
                    response_generated = True
            with col2:
                if st.button("🔑 Key Facts", key="facts"):
                    response_type = "Key Facts"
                    response_generated = True
            with col3:
                if st.button("📚 Research", key="research"):
                    response_type = "Research Papers"
                    response_generated = True
            with col4:
                if st.button("💡 Simple", key="simple"):
                    response_type = "Simple Explanation"
                    response_generated = True
            
            if response_generated:
                compound_name = st.session_state.current_molecule['name']
                prompt = PROMPT_TEMPLATES[response_type].format(compound=compound_name)
                
                with st.spinner(f"🦙 Llama 3 is analyzing {compound_name}..."):
                    response, ai_error = generate_ai_response(tokenizer, model, prompt)
                
                if ai_error:
                    st.error(f"❌ AI generation failed: {ai_error}")
                    
                    # Provide fallback options
                    st.info("💡 **Alternative options:**")
                    chatgpt_link = create_chatgpt_link(compound_name)
                    st.markdown(f"- [Ask ChatGPT about {compound_name}]({chatgpt_link})")
                    st.markdown(f"- [Search PubMed for {compound_name}](https://pubmed.ncbi.nlm.nih.gov/?term={urllib.parse.quote(compound_name + ' blood brain barrier')})")
                    
                elif response:
                    st.markdown(f"**{response_type}:**")
                    st.markdown(f"> {response}")
                    
                    # Show API usage info
                    st.caption("🦙 Generated by Llama 3 8B Instruct via Hugging Face API")
                    
                    if len(response) < 20:
                        st.warning("⚠️ Short response detected. The model might be warming up. Try again or use alternative options below.")
                else:
                    st.warning("⚠️ No response generated. Try a different style or check your API connection.")
            
            st.markdown("---")
            st.markdown("### 🔍 Continue Your Research")
            
            col1, col2 = st.columns(2)
            with col1:
                chatgpt_link = create_chatgpt_link(st.session_state.current_molecule['name'])
                st.markdown(f"[💬 ChatGPT Analysis]({chatgpt_link})")
            
            with col2:
                pubmed_link = f"https://pubmed.ncbi.nlm.nih.gov/?term={urllib.parse.quote(st.session_state.current_molecule['name'] + ' blood brain barrier')}"
                st.markdown(f"[📚 PubMed Search]({pubmed_link})")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.caption("NeuroGate v2025.01 | Advanced BBB Prediction with Uncertainty Quantification")

print(f"Python executable: {sys.executable}")