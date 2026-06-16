"""
setup_features.py

One-time script to extract PaDEL descriptor feature names for model alignment.
- Generates descriptors for reference molecule (aspirin: CC(=O)OC1=CC=CC=C1C(=O)O)
- Loads a trained model to verify feature count
- Saves feature names to output/models/feature_names.pkl
"""
import os
import joblib
import pandas as pd
import logging
from padelpy import from_smiles

REFERENCE_SMILES = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
FEATURES_OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../output/models/feature_names.txt'))
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../output/models/padel_KNN_model.pkl'))  # Use any PaDEL model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("setup_features")

def main():
    logger.info("Generating PaDEL descriptors for reference molecule: %s", REFERENCE_SMILES)
    descriptors = from_smiles(REFERENCE_SMILES, fingerprints=False, descriptors=True)
    df = pd.DataFrame([descriptors])
    if 'Name' in df.columns:
        df = df.drop(columns=['Name'])
    feature_names = list(df.columns)
    logger.info("Extracted %d descriptor features.", len(feature_names))

    logger.info("Loading model from: %s", MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    n_model_features = getattr(model, 'n_features_in_', None)
    logger.info("Model expects %s features.", n_model_features)

    if n_model_features is not None and n_model_features != len(feature_names):
        logger.warning("Feature count mismatch: model expects %d, descriptors found %d", n_model_features, len(feature_names))
    else:
        logger.info("Feature count matches.")

    logger.info("Saving feature names to: %s", FEATURES_OUT)
    os.makedirs(os.path.dirname(FEATURES_OUT), exist_ok=True)
    joblib.dump(feature_names, FEATURES_OUT)
    logger.info("Done.")


if __name__ == "__main__":
    main()
    
