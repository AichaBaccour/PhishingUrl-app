from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(
    title="🛡️ Phishing Detection API",
    description="API FastAPI pour la prédiction en temps réel du risque de phishing sur le dataset PhiUSIIL.",
    version="1.0.0"
)

# Chargement du modèle sauvegardé
try:
    model = joblib.load("models/best_model.joblib")
except Exception:
    model = None

# Schéma d'entrée des données
class FeaturesInput(BaseModel):
    features: list[float]

@app.get("/")
def root():
    return {
        "status": "online",
        "message": "API de Détection de Phishing opérationnelle.",
        "documentation": "/docs"
    }

@app.post("/predict")
def predict(data: FeaturesInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé sur le serveur.")
    
    # Formatage de l'entrée sous forme de tableau 2D NumPy
    X_input = np.array(data.features).reshape(1, -1)
    
    # Calcul des probabilités
    proba_phishing = float(model.predict_proba(X_input)[0][1])
    prediction = int(proba_phishing >= 0.50)
    
    # Détermination du niveau de risque métier
    if proba_phishing >= 0.70:
        risk_level = "Élevé 🔴"
    elif proba_phishing >= 0.35:
        risk_level = "Moyen 🟠"
    else:
        risk_level = "Faible 🟢"

    return {
        "prediction": prediction,
        "label": "Phishing" if prediction == 1 else "Légitime",
        "proba_phishing": round(proba_phishing, 4),
        "proba_legitime": round(1.0 - proba_phishing, 4),
        "niveau_risque": risk_level,
        "confiance_pourcentage": round(max(proba_phishing, 1.0 - proba_phishing) * 100, 2)
    }