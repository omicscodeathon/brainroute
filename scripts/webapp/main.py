# app.py
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

import requests


from api import get_smiles
# everything here right now - need to get modular
# bbb_app 
# -main.py (landing page)
# -backend.py (take input from main and run models)
# -LLM/CHEMBL api - connected to main 


#main - takes in smiles or compound name - model prediction using labels so if name given - function to get smiles for it 



# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="BBB Predictor",
    page_icon="🧪",
    layout="centered"
)



st.markdown("<h1 style='text-align: center;'> 🧠 NeuroGate!!! </h1>", unsafe_allow_html=True)
# -------------------------
# Landing Page
# -------------------------
#st.title("🧠 NeuroGate")
st.subheader("Blood-Brain Barrier Penetration Classifier", width = "stretch")

st.markdown(
    """
    This tool allows you to explore molecules and predict their **Blood-Brain Barrier (BBB) penetration**.
    
    - Enter a **compound name** (e.g., aspirin) or  
    - Provide a **SMILES string** (e.g., `CC(=O)OC1=CC=CC=C1C(=O)O`)  
    
    Neurogate will classify your molecule as **BBB+ or BBB-** and provide additional information about the molecule for further drug discovery research! 
    """
)

# -------------------------
# Input Section
# -------------------------
input_mode = st.radio(
    "How would you like to provide your compound?",
    ["Compound Name", "SMILES String"]
)

user_input = st.text_input(
    f"Enter {input_mode}:",
    placeholder="e.g., aspirin OR CC(=O)OC1=CC=CC=C1C(=O)O"
)



        
if st.button("Submit"):
    if user_input.strip() == "":
        st.warning("Please enter a compound name or SMILES string.")
    else:
        st.success(f" You entered: {user_input}")

        # Try to render molecule if SMILES
        #todo - input modes not required - check automatically what given 
        if input_mode == "SMILES String":
            try:
                mol = Chem.MolFromSmiles(user_input)
                if mol:
                    st.image(Draw.MolToImage(mol), caption="Molecule Structure")
                else:
                    st.error("Invalid SMILES string.")
            except Exception as e:
                st.error(f"Error laoding: {e}")
        
        elif input_mode == "Compound Name":
            smiles = get_smiles(user_input)
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    st.image(Draw.MolToImage(mol), caption="Molecule Structure")
                else:
                    st.error("Couldn't retreive smiles")
            except Exception as e:
                st.error(f"Error loading:  {e}")
            


#TODO - info about what molecule type, max phase, mechanism of action, therapeutic indication means from chembl 

# Footer

st.markdown("---")
st.caption("Neurogate - Work in progress")
