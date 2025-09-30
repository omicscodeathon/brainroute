# app.py
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, Crippen, Lipinski
import requests
import re
import pandas as pd
import urllib.parse


import sys
import os 
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from api import get_smiles
from api import get_chembl_info
from models import XGB_model, KNN_model, scaler


# model_names = ['KNN', 'SVM', 'RF', 'LR', 'XGB']
# loaded_models = {name: joblib.load(f'../../output/models/{name}_model.pkl') for name in model_names }
# MLP_model = tf.keras.models.load_model('../../output/models/MLP_model.h5')


#todo fixes -
#when selecting diff options - only loads AI output - initially loaded not visible 


#major todos - 
#connect models here - implement use all models and give best one through consensus everytime (if not too slow)
#use api tokens to use online version - this script can stay on github for local implementation
#deploy on streamlit community - or huggingface if that's easier 



# everything here right now - need to get modular
# bbb_app 
# -main.py (landing page)
# -backend.py (take input from main and run models)
# -LLM/CHEMBL api - connected to main 


#main - takes in smiles or compound name - model prediction using labels so if name given - function to get smiles for it 

def ensure_complete_sentence(text):
    """
    Ensures the text ends with a complete sentence by finding the last proper sentence ending.
    """
    if not text.strip():
        return text
    
    # Common sentence endings
    sentence_endings = ['.', '!', '?']
    
    # Find the last occurrence of any sentence ending
    last_ending_pos = -1
    for ending in sentence_endings:
        pos = text.rfind(ending)
        if pos > last_ending_pos:
            last_ending_pos = pos
    
    if last_ending_pos == -1:
        # No sentence ending found, add a period
        return text.strip() + "."
    else:
        # Return text up to and including the last sentence ending
        return text[:last_ending_pos + 1].strip()

def create_chatgpt_link(molecule_name):
    """
    Creates a ChatGPT link with a pre-populated prompt about the molecule.
    """
    prompt = f"Tell me more about {molecule_name} and its relevance in CNS drug discovery, with context of blood-brain barrier (BBB) penetration."
    encoded_prompt = urllib.parse.quote(prompt)
    return f"https://chat.openai.com/?q={encoded_prompt}"

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="BBB Predictor",
    page_icon="🧪",
    layout="wide"
)

print(sys.executable)

st.markdown("<h1 style='text-align: center;'> 🧠 NeuroGate </h1>", unsafe_allow_html=True)
# -------------------------
# Landing Page
# -------------------------
#st.title("🧠 NeuroGate")
st.subheader("Blood-Brain Barrier Penetration Classifier")

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

#ability to handle scientific names - aspirin is 2-Acetoxybenzoic acid or Acetylsalicylic acid

#pressing enter should also work 
# --- Submit Action ---
if st.button("Submit"):
    if user_input.strip() == "":
        st.warning("Please enter a compound name or SMILES string.")
    else:
        st.success(f"You entered: {user_input}")

        smiles = None
        mol = None
        
        info = get_chembl_info(user_input)

        if input_mode == "SMILES String":
            try:
                mol = Chem.MolFromSmiles(user_input)
                smiles = user_input if mol else None
            except Exception as e:
                st.error(f"Error loading: {e}")

        elif input_mode == "Compound Name":
            # Placeholder: Replace with actual lookup function
            try:
                smiles = get_smiles(user_input)
                mol = Chem.MolFromSmiles(smiles) if smiles else None
            except Exception as e:
                st.error(f"Error loading: {e}")
                
                
        if mol:
            descr = pd.DataFrame([Descriptors.CalcMolDescriptors(mol)])
            
            scale = scaler()
            descr = scale.transform(descr)
            # xgb = XGB_model()
            knn = KNN_model()
            # pred = xgb.predict(descr)
            pred = knn.predict(descr)
            # confidence = xgb.predict_proba(descr)
            confidence = knn.predict_proba(descr)
            
            bbb_prediction = "BBB+" if pred[0] == 1 else "BBB-"
            conf_score = confidence[0][pred[0]] * 100
            

            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)

            # #test prediction - replace with model implementation 
            # bbb_prediction = "BBB+" if logp > 2 and mw < 450 else "BBB-"

            # --- Layout: Two Columns ---
            col1, col2 = st.columns([1, 1.5])

            with col1:
                st.image(Draw.MolToImage(mol), caption="Molecule Structure")
                #get 3d structure?  - or provide option to 

            with col2:
                st.markdown("### 🧪 Molecule Properties")
                st.markdown(f"**SMILES:** `{smiles}`")
                st.markdown(f"**Molecular Weight:** {mw:.2f}")
                st.markdown(f"**LogP:** {logp:.2f}")
                st.markdown(f"**H-Bond Donors:** {hbd}")
                st.markdown(f"**H-Bond Acceptors:** {hba}")
                st.markdown(f"**BBB Prediction:** :blue[{bbb_prediction}]")
                st.markdown(f"**Confidence Score:** {conf_score:.2f}%")
                #test 
                if info:
                    st.markdown(f"**Molecule Type: {info['Molecule Type']}**")
                    st.markdown(f"**Max Phase : {info['Max Phase']}**")
                    st.markdown(f"**Mechanism of Action: {info['Mechanism of Action']}**")
                    st.markdown(f"**Therapeutic Indications: {info['Therapeutic Indications']}**")
        
        
            st.markdown(
            """
            LogP - Octanol-water partition coefficient \n
            Molecule Type - classification of a molecule based on its chemical nature and biological role \n
            Max phase - the maximum phase of clinical development a drug or compound has achieved for any of its therapeutic uses \n
            Mechanism of Action - associating drugs and chemical compounds with their specific biological targets and the observed effects of that interaction \n
            Therapeutic Indications - specific diseases or conditions that a drug is intended to treat or alleviate \n
            """
            )
            


            
#LLM implementation 

st.subheader("🔍 AI Overview")

if "prompt" not in st.session_state:
    st.session_state.prompt = None

import torch
#from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

if user_input:
    with st.spinner("Loading AI model and generating response..."):
        model_name = "microsoft/biogpt"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        
        set_seed = 24

        molecule_name = user_input

        #diff prompt stules 
        # prompt_option = st.selectbox(
        #     "Choose AI response style:",
        #     ["Summary", "Key Facts", "Research Papers", "Simple Explanation"],
        #     index=0
        # )
        st.markdown("Choose response style - ")
        

        if st.button("Summary"):
            prompt_option = "Summary"
            st.session_state.prompt = f"{user_input} is a compound that"
        elif st.button("Key Facts"):
            prompt_option = "Key Facts"
            st.session_state.prompt = f"Key facts about {user_input} and blood-brain barrier penetration: 1."
        elif st.button("Research Papers"):
            prompt_option = "Research Papers"
            st.session_state.prompt = f"Important research papers on {user_input} and BBB include:"
        elif st.button("Simple Explanation"):
            prompt_option = "Simple Explanation"
            st.session_state.prompt = f"{user_input} affects the brain because it"
        
        prompt = st.session_state.prompt
        
        if st.session_state.prompt:
            inputs = tokenizer(prompt, return_tensors="pt")

            device = torch.device("cpu")
            model.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=300,  # Shorter for more coherent output
                    min_length=len(inputs['input_ids'][0]) + 50,  # Shorter minimum
                    temperature=0.5,  # Lower temperature for more focused output
                    do_sample=True, 
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    no_repeat_ngram_size=2,  # Prevent repetition
                    early_stopping=True,
                    num_beams=3,  # Use beam search for better quality
                    top_p=0.8,  # More focused sampling
                    repetition_penalty=1.3,  # Higher penalty for repetition
                    length_penalty=1.0,  # Neutral length penalty
                    max_new_tokens=150,  # Limit new content
                )

            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = full_output[len(prompt):].strip()

            # Clean up the generated text
            def clean_generated_text(text):
                # Remove common artifacts and weird formatting
                text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                text = re.sub(r'\.{2,}', '.', text)  # Fix multiple periods
                text = re.sub(r'\([^)]*\)', '', text)  # Remove parenthetical citations
                text = re.sub(r'ABSTRACT.*?WORDS\)', '', text, flags=re.IGNORECASE)
                text = re.sub(r'LEARNING OBJECTIVE.*?:', '', text, flags=re.IGNORECASE)
                text = re.sub(r'DATA SOURCES.*?:', '', text, flags=re.IGNORECASE)
                text = re.sub(r'CONCLUSIONS?:', '', text, flags=re.IGNORECASE)
                return text.strip()

            # Clean and ensure complete sentence
            cleaned_text = clean_generated_text(generated_text)
            complete_text = ensure_complete_sentence(cleaned_text)

            # Display based on selected option
            if prompt_option == "Research Papers":
                st.markdown("### 📚 Related Research")
                if complete_text:
                    st.markdown(f"**Studies on {molecule_name} and BBB:** {complete_text}")
                else:
                    st.info("No specific research papers found in the model's knowledge base.")
            else:
                st.markdown(f"**{prompt_option}:** {complete_text}")

            # Fallback option if output is still poor
            if len(complete_text) < 20 or "ABSTRACT" in complete_text.upper():
                st.warning("⚠️ AI model output may be incoherent. Try a different response style or use the ChatGPT link below for better results.")

        # Continue Research Section
        st.markdown("---")
        st.markdown("### 🔍 Continue Your Research")
        
        chatgpt_link = create_chatgpt_link(user_input)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"[💬 Ask ChatGPT]({chatgpt_link})")
        with col2:
            st.caption("Get more detailed insights about this molecule's CNS drug development potential")

        with st.expander("View Original Prompt"):
            st.text(prompt)



# output_human = tokenizer.decode(outputs[0], skip_special_tokens=True)
# st.markdown(output_human)


                
#TODO - when error - ask chatgpt button - with error preloaded - can exploit? 


#giving intel-m1 issue 
#from transformers import pipeline

#TODO - figure out tokens - the appropriate max_limit for a 

#generator = pipeline("text-generation", model="google/flan-t5-small", device="mps")

# generator = pipeline("text2text-generation", model="gpt2-medium", device="mps")

# #not using m1 gpu right now - super slow 
# #TODO - get it running locally 
# #try different models and find one that works best 


# molecule_name = user_input

# if molecule_name:
#     prompt = f"tell me more about {molecule_name} in 200 characters"
#     result = generator(prompt, max_length=200, num_return_sequences=1)
#     st.write(result[0]['generated_text'])

# pipe = pipeline(
#     "text-generation",
#     model="gpt2",
#     max_new_tokens=100,   # generate 100 tokens
#     do_sample=True,       # for stochastic generation
#     temperature=0.7       # creativity control
# )


#TODO - info about what molecule type, max phase, mechanism of action, therapeutic indication means from chembl 
#TODO - work on a better loading sign 




# Footer

st.markdown("---")
st.caption("Neurogate - v2025.00")