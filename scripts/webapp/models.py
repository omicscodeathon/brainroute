import joblib
from tensorflow.keras import models 

def KNN_model():
    model = joblib.load('output/models/KNN_model.pkl')
    return model

def SVM_model():
    model = joblib.load('output/models/SVM_model.pkl')
    return model

def RF_model():
    model = joblib.load('output/models/RF_model.pkl')
    return model

def LR_model():
    model = joblib.load('output/models/LR_model.pkl')
    return model

def XGB_model():
    model = joblib.load('output/models/XGB_model.pkl')
    return model

def MLP_model():
    model = models.load_model('output\models\MLP_model.h5')
    return model

def scaler():
    scaler = joblib.load('output/models/scaler.pkl')
    return scaler
