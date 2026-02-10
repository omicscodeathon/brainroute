import streamlit as st


def _handle_nav_query(current: str) -> None:
    nav = None
    try:
        nav = st.query_params.get("nav")
        if isinstance(nav, list):
            nav = nav[0] if nav else None
    except Exception:
        try:
            nav = st.experimental_get_query_params().get("nav", [None])[0]
        except Exception:
            nav = None

    if not nav:
        return

    nav = str(nav).lower().strip()

    # Remove only our navigation parameter to avoid clobbering any other query state.
    try:
        if "nav" in st.query_params:
            del st.query_params["nav"]
    except Exception:
        try:
            qp = st.experimental_get_query_params()
            qp.pop("nav", None)
            st.experimental_set_query_params(**qp)
        except Exception:
            pass

    if nav == current:
        return

    if nav == "home":
        st.switch_page("main.py")
    elif nav == "tutorial":
        st.switch_page("pages/tutorial.py")
    elif nav == "about":
        st.switch_page("pages/about.py")

st.set_page_config(
    page_title="About | BrainRoute",
    page_icon=None,
    layout="wide"
)

_handle_nav_query(current="about")

# CSS - sky blue + glassmorphism blue theme
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
    
    /* Force sans-serif and dark text globally */
    *, html, body, [class*="st-"], p, span, div, label, h1, h2, h3, h4, h5, h6, li, td, th {
        font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif !important;
        color: #0b1b3a !important;
    }
    
    /* Force light sky blue background */
    .stApp, .main, [data-testid="stAppViewContainer"] {
        background-color: #eef6ff !important;
    }
    
    .main .block-container {
        max-width: 900px;
        padding: 3.5rem 1rem 2rem 1rem;
    }
    
    /* Fixed top navigation ribbon */
    .nav-ribbon {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem 2.5rem;
        background: rgba(31, 78, 153, 0.35);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-bottom: 1px solid rgba(59, 130, 246, 0.3);
        box-shadow: 0 2px 16px rgba(31, 78, 153, 0.15);
    }
    
    .nav-ribbon .nav-left,
    .nav-ribbon .nav-right {
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }
    
    .nav-ribbon a {
        color: #0b1b3a !important;
        text-decoration: none !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.4rem 0 !important;
        background: none !important;
        border: none !important;
        transition: all 0.2s ease !important;
        display: inline-block !important;
    }
    
    .nav-ribbon a:hover {
        color: #1f4e99 !important;
    }
    
    .nav-ribbon .nav-brand {
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.02em;
    }
    
    /* Header with glassmorphism */
    .about-header {
        background: linear-gradient(135deg, rgba(31, 78, 153, 0.9) 0%, rgba(45, 107, 200, 0.9) 50%, rgba(24, 64, 128, 0.9) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(59, 130, 246, 0.7);
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 2.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.3);
    }
    
    .about-header h1 {
        font-size: 2.2rem;
        font-weight: bold;
        color: #f5f9ff !important;
        margin: 0 0 0.5rem 0;
    }
    
    .about-header p {
        font-size: 1rem;
        color: #f5f9ff !important;
        margin: 0;
    }
    
    /* Section cards */
    .section-card {
        background: linear-gradient(135deg, rgba(31, 78, 153, 0.92) 0%, rgba(45, 107, 200, 0.92) 100%);
        border: 1px solid rgba(59, 130, 246, 0.7);
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    
    .section-card h2 {
        font-size: 1.3rem;
        font-weight: bold;
        color: #f5f9ff !important;
        margin: 0 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #3b82f6;
    }
    
    .section-card h3 {
        font-size: 1.1rem;
        font-weight: bold;
        color: #f5f9ff !important;
        margin: 1.25rem 0 0.75rem 0;
    }
    
    .section-card p, .section-card li {
        font-size: 0.95rem;
        color: #f5f9ff !important;
        line-height: 1.7;
    }
    
    .section-card ul {
        padding-left: 1.25rem;
        margin: 0.5rem 0;
    }
    
    .section-card li {
        margin-bottom: 0.5rem;
    }
    
    .section-card a {
        color: #93c5fd !important;
        text-decoration: underline !important;
    }
    
    .section-card a:hover {
        color: #bfdbfe !important;
    }
    
    /* Footer */
    .app-footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1.5rem;
        color: #0b1b3a !important;
        font-size: 0.85rem;
        border-top: 1px solid #cfe6ff;
    }
</style>
""", unsafe_allow_html=True)

# Fixed top navigation ribbon
st.markdown('''
<div class="nav-ribbon">
    <div class="nav-left">
        <a href="?nav=home" target="_self" class="nav-brand">BrainRoute</a>
    </div>
    <div class="nav-right">
        <a href="?nav=home" target="_self">Home</a>
        <a href="?nav=tutorial" target="_self">Tutorial</a>
        <a href="?nav=about" target="_self">About</a>
        <a href="https://omicscodeathon.github.io/brainroutedb" target="_blank">Database ↗</a>
    </div>
</div>
''', unsafe_allow_html=True)

# Header
st.markdown("""
<div class="about-header">
    <h1>About BrainRoute</h1>
    <p>Blood-Brain Barrier penetration prediction for drug discovery</p>
</div>
""", unsafe_allow_html=True)

# Project Overview
st.markdown("""
<div class="section-card">
    <h2>Project Overview</h2>
    <p>
        BrainRoute is a machine-learning-powered web application that predicts whether a given molecule can cross the 
        <strong>Blood-Brain Barrier (BBB)</strong>. The BBB is a highly selective semipermeable membrane that separates 
        circulating blood from the brain's extracellular fluid, protecting the central nervous system from toxins, 
        pathogens, and other harmful agents. Understanding BBB permeability is a critical step in developing drugs that 
        target neurological disorders such as Alzheimer's disease, Parkinson's disease, epilepsy, and brain cancers.
    </p>
    <p>
        Given a molecule — entered as a compound name or SMILES string — BrainRoute classifies it as 
        <strong>BBB+</strong> (likely to penetrate) or <strong>BBB-</strong> (unlikely to penetrate), along with 
        confidence scores and key molecular properties that influence permeability.
    </p>
</div>
""", unsafe_allow_html=True)

# How it Works
st.markdown("""
<div class="section-card">
    <h2>How It Works</h2>
    <h3>Ensemble Machine Learning</h3>
    <p>
        BrainRoute uses an ensemble of three independently trained classifiers that vote on the final prediction:
    </p>
    <ul>
        <li><strong>K-Nearest Neighbors (KNN)</strong> — classifies molecules based on similarity to known compounds in the training set.</li>
        <li><strong>LightGBM (LGBM)</strong> — a gradient-boosted decision tree framework optimised for speed and accuracy.</li>
        <li><strong>Extra Trees (ET)</strong> — a randomized ensemble of decision trees that reduces variance and overfitting.</li>
    </ul>
    <p>
        The final BBB+/BBB- call is determined by majority voting. Each model also outputs a probability, and these 
        are averaged to produce an overall confidence score.
    </p>
    <h3>PaDEL Molecular Descriptors</h3>
    <p>
        Molecules are featurised using <strong>PaDEL-Descriptor</strong>, which computes over 1,400 one- and two-dimensional 
        molecular descriptors (topological, constitutional, electronic, etc.) from SMILES. These descriptors serve as the 
        input features for the ML models.
    </p>
    <h3>AI Chat</h3>
    <p>
        After obtaining a prediction, users can chat with <strong>Llama 3</strong> (via the Hugging Face Inference API) to 
        ask follow-up questions about the molecule's pharmacology, side effects, research literature, and drug potential.
    </p>
</div>
""", unsafe_allow_html=True)

# Training Data
st.markdown("""
<div class="section-card">
    <h2>Training Data</h2>
    <p>
        The models were trained on the <strong>B3DB (Blood-Brain Barrier Database)</strong>, a curated open-source dataset 
        containing experimentally determined BBB permeability labels for thousands of small molecules. Compound information 
        is cross-referenced against <strong>ChEMBL</strong> and <strong>PubChem</strong> to ensure structural accuracy.
    </p>
</div>
""", unsafe_allow_html=True)

# Tech Stack
st.markdown("""
<div class="section-card">
    <h2>Technology Stack</h2>
    <ul>
        <li><strong>Frontend:</strong> Streamlit</li>
        <li><strong>ML Models:</strong> scikit-learn, LightGBM</li>
        <li><strong>Chemistry:</strong> RDKit, PaDEL-Descriptor</li>
        <li><strong>AI Chat:</strong> Meta Llama 3 via Hugging Face Inference API</li>
        <li><strong>Database:</strong> Neon PostgreSQL (BrainRouteDB)</li>
        <li><strong>Visualization:</strong> Plotly</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Team & Links
st.markdown("""
<div class="section-card">
    <h2>Team & Links</h2>
    <p>
        BrainRoute was developed as part of the <strong>Omics-Codeathon</strong> project — an initiative to build 
        open-source bioinformatics and cheminformatics tools.
    </p>
    <ul>
        <li><a href="https://github.com/omicscodeathon/brainroute" target="_blank">GitHub Repository</a></li>
        <li><a href="https://omicscodeathon.github.io/brainroutedb" target="_blank">BrainRoute Database</a></li>
    </ul>
    <p>
        For bug reports, feature requests, or collaboration inquiries, please open an issue on the GitHub repository.
    </p>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="app-footer">
    BrainRoute v2026.01 | Omics-Codeathon
</div>
""", unsafe_allow_html=True)
