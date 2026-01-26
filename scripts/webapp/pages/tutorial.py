import streamlit as st

st.set_page_config(
    page_title="Tutorial & Info | BrainRoute",
    page_icon=None,
    layout="wide"
)

# Minimal CSS - Times New Roman, black text, works in light/dark modes
st.markdown("""
<style>
    /* Hide sidebar completely */
    [data-testid="stSidebar"] {display: none;}
    [data-testid="stSidebarNav"] {display: none;}
    section[data-testid="stSidebar"] {display: none;}
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Force Times New Roman and black text globally */
    *, html, body, [class*="st-"], p, span, div, label, h1, h2, h3, h4, h5, h6, li, td, th {
        font-family: 'Times New Roman', Times, serif !important;
        color: #000000 !important;
    }
    
    /* Force white background */
    .stApp, .main, [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
    }
    
    .main .block-container {
        max-width: 900px;
        padding: 2rem 1rem;
    }
    
    /* Floating Home button */
    .floating-nav {
        position: fixed;
        top: 1rem;
        left: 1.5rem;
        z-index: 1000;
    }
    
    /* Style Streamlit button to match design */
    .stButton > button {
        background: #ffffff !important;
        border: 2px solid #000000 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.2rem !important;
        font-size: 0.9rem !important;
        font-weight: normal !important;
        color: #000000 !important;
        font-family: 'Times New Roman', Times, serif !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        background: #000000 !important;
        color: #ffffff !important;
        border-color: #000000 !important;
    }
    
    /* Header with glassmorphism */
    .tutorial-header {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 2.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .tutorial-header h1 {
        font-size: 2.2rem;
        font-weight: bold;
        color: #000000 !important;
        margin: 0 0 0.5rem 0;
    }
    
    .tutorial-header p {
        font-size: 1rem;
        color: #000000 !important;
        margin: 0;
    }
    
    /* Section cards */
    .section-card {
        background: #f8f8f8;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 1.5rem;
    }
    
    .section-card h2 {
        font-size: 1.3rem;
        font-weight: bold;
        color: #000000 !important;
        margin: 0 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .section-card h3 {
        font-size: 1.1rem;
        font-weight: bold;
        color: #000000 !important;
        margin: 1.25rem 0 0.75rem 0;
    }
    
    .section-card p, .section-card li {
        font-size: 0.95rem;
        color: #000000 !important;
        line-height: 1.7;
    }
    
    .section-card ul {
        padding-left: 1.25rem;
        margin: 0.5rem 0;
    }
    
    .section-card li {
        margin-bottom: 0.5rem;
    }
    
    .placeholder-note {
        background: #f0f0f0;
        border-left: 3px solid #000000;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #000000 !important;
        font-style: italic;
    }
    
    /* Inline code styling */
    code {
        background-color: #f0f0f0 !important;
        color: #000000 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        font-family: 'Courier New', monospace !important;
    }
    
    /* Footer */
    .app-footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1.5rem;
        color: #000000 !important;
        font-size: 0.85rem;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Floating Home button container
st.markdown('<div class="floating-nav">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 10])
with col1:
    if st.button("Home", key="home_btn", type="secondary"):
        st.switch_page("main.py")
st.markdown('</div>', unsafe_allow_html=True)

# Header
st.markdown("""
<div class="tutorial-header">
    <h1>Tutorial & Information</h1>
    <p>Learn how to use BrainRoute for BBB penetration prediction</p>
</div>
""", unsafe_allow_html=True)

# Usage Instructions Section
st.markdown("---")
st.markdown("## Usage Instructions")

st.markdown("### Single Molecule Analysis")
st.markdown("""
To analyze a single molecule:
- **Step 1:** Select "Single Molecule" from the Processing Mode options
- **Step 2:** Choose your input method - either "Compound Name" or "SMILES String"
- **Step 3:** Enter your molecule identifier in the text field
- **Step 4:** Click "Analyze Molecule" to run the prediction

Results will display the BBB penetration prediction (BBB+ or BBB-), confidence scores from three ML models (KNN, LGBM, ET), molecular structure, and key physicochemical properties.
""")

st.markdown("### Batch Processing")
st.markdown("""
For analyzing multiple molecules at once:
- **CSV Upload:** Upload a CSV file with columns named "smiles" and/or "name". The system will process each row and return predictions for all molecules.
- **Text Input:** Enter one molecule per line (either SMILES strings or compound names). Press Enter after each molecule.

After processing, you can view summary statistics, filter results, and export to CSV, Excel, or JSON formats.
""")

st.markdown("### AI Chat Feature")
st.markdown("""
After analyzing a molecule in Single Molecule mode, you can chat with Llama 3 about your results:
- Click "Connect to Llama 3" to initialize the AI assistant
- Use the Quick Question buttons for common queries about drug potential, molecular properties, side effects, or research
- Type custom questions in the chat input field for detailed pharmacological insights
- Export your chat history for documentation purposes
""")

# Requirements Section
st.markdown("---")
st.markdown("## Application Requirements")

st.markdown("### System Requirements")
st.markdown("""
- **Browser:** Modern web browser (Chrome, Firefox, Safari, Edge) with JavaScript enabled
- **Internet:** Stable internet connection required for compound lookups and AI chat features
- **Display:** Minimum 1280x720 resolution recommended for optimal viewing
""")

st.markdown("### Input Formats")
st.markdown("""
**SMILES Strings:**
- Use valid SMILES (Simplified Molecular Input Line Entry System) notation
- Example: `CC(=O)OC1=CC=CC=C1C(=O)O` (aspirin)
- Canonical SMILES preferred for consistency
- Avoid salts and counterions when possible - use parent compound

**Compound Names:**
- Use scientific/IUPAC names or common drug names registered in PubChem/ChEMBL
- Examples: donepezil, rivastigmine, memantine, galantamine
- Avoid brand names - use generic names instead
- For complex molecules, SMILES input may be more reliable
""")

st.markdown("### CSV Format for Batch Processing")
st.markdown("""
Your CSV file should include at least one of these columns:
- **smiles** - Column containing SMILES strings (recommended)
- **name** - Column containing compound names

If both columns are present, SMILES will be used preferentially. Additional columns will be preserved in the output.
""")

# General Information Section
st.markdown("---")
st.markdown("## General Information")

st.markdown("### About BrainRoute")
st.markdown("""
BrainRoute is a Blood-Brain Barrier (BBB) penetration prediction tool designed for drug discovery research. 
The BBB is a selective barrier that protects the brain from potentially harmful substances while allowing essential nutrients to pass through. 
For CNS (Central Nervous System) drugs, predicting BBB penetration is crucial - the drug must cross this barrier to reach its target in the brain.

BrainRoute classifies molecules as:
- **BBB+** - Predicted to penetrate the blood-brain barrier
- **BBB-** - Predicted to NOT penetrate the blood-brain barrier
""")

st.markdown("### Model Information")
st.markdown("""
BrainRoute uses an ensemble of three machine learning models trained on PaDEL molecular descriptors:
- **KNN (K-Nearest Neighbors):** Instance-based learning algorithm that classifies based on similarity to training examples
- **LGBM (Light Gradient Boosting Machine):** Gradient boosting framework using tree-based learning algorithms
- **ET (Extra Trees):** Ensemble method using randomized decision trees for robust predictions

The final prediction is determined by majority voting across all three models. Confidence scores reflect the probability estimates from each classifier.
""")

st.markdown("### Molecular Properties Displayed")
st.markdown("""
- **Molecular Weight (MW):** Total mass of the molecule. Generally, MW < 450 Da favors BBB penetration
- **LogP:** Partition coefficient indicating lipophilicity. Optimal range for BBB: 1-3
- **H-Bond Donors (HBD):** Number of hydrogen bond donors. Fewer donors (<3) favor BBB penetration
- **H-Bond Acceptors (HBA):** Number of hydrogen bond acceptors. Fewer acceptors (<7) favor BBB penetration
- **TPSA:** Topological Polar Surface Area. Lower TPSA (<90 A squared) generally indicates better BBB permeability
- **Rotatable Bonds:** Measure of molecular flexibility. Fewer rotatable bonds may improve BBB penetration
""")

st.markdown("### Data Sources")
st.markdown("""
- **Training Data:** B3DB (Blood-Brain Barrier Database) - curated dataset of BBB permeability measurements
- **Compound Information:** ChEMBL and PubChem databases for molecular structure validation and metadata
- **Molecular Descriptors:** PaDEL-Descriptor software for calculating 2D/3D molecular descriptors
""")

st.markdown("### Limitations & Disclaimers")
st.markdown("""
- Predictions are computational estimates and should be validated experimentally
- Model accuracy depends on the chemical space of training data - novel scaffolds may have higher uncertainty
- BBB penetration in vivo depends on many factors not captured by molecular descriptors (efflux transporters, metabolism, etc.)
- This tool is for research purposes only and not intended for clinical decision-making
""")

st.markdown("### Contact & Support")
st.markdown("""
BrainRoute was developed as part of the Omics-Codeathon project.
- **GitHub:** [github.com/omicscodeathon](https://github.com/omicscodeathon)
- **Database:** [BrainRoute Database](https://omicscodeathon.github.io/brainroutedb)

For bug reports, feature requests, or collaboration inquiries, please open an issue on our GitHub repository.
""")

# Tips Section
st.markdown("---")
st.markdown("## Tips for Best Results")

st.markdown("### Input Recommendations")
st.markdown("""
- **Use SMILES when possible:** SMILES strings provide unambiguous molecular identification and avoid name lookup failures
- **Standardize structures:** Use parent compounds without salts, remove stereochemistry if not essential
- **Check compound names:** Verify your compound name exists in PubChem or ChEMBL before submitting
- **Batch processing:** For large datasets, break into batches of 50-100 molecules for optimal performance
""")

st.markdown("### Interpreting Results")
st.markdown("""
- **High confidence (>80%):** Models agree strongly - prediction is more reliable
- **Medium confidence (60-80%):** Some model disagreement - consider additional validation
- **Low confidence (<60%):** High uncertainty - experimental validation strongly recommended
- **Model agreement:** When all 3 models agree, the prediction is more trustworthy
""")

st.markdown("### Example Compounds to Try")
st.markdown("""
- **Known BBB+ compounds:** caffeine, nicotine, diazepam, donepezil
- **Known BBB- compounds:** dopamine, serotonin, atenolol, metformin
""")

# Footer
st.markdown("""
<div class="app-footer">
    BrainRoute v2025.01 | Omics-Codeathon
</div>
""", unsafe_allow_html=True)
