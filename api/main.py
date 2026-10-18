from pathlib import Path

import joblib
import pandas as pd
import scipy.sparse as sp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="🛡️ Phishing Detection API",
    description="API FastAPI de prédiction du risque de phishing (dataset PhiUSIIL).",
    version="2.0.0",
)

FEATURES = [
    "URLLength", "DomainLength", "IsDomainIP", "CharContinuationRate",
    "TLDLegitimateProb", "URLCharProb", "TLDLength", "NoOfSubDomain",
    "HasObfuscation", "ObfuscationRatio", "LetterRatioInURL", "DegitRatioInURL",
    "SpacialCharRatioInURL", "NoOfURLRedirect", "NoOfSelfRedirect",
]
BASE_DIR = Path(__file__).resolve().parent.parent


def load_artifact(name):
    for path in (BASE_DIR / "models" / name, Path("models") / name):
        if path.exists():
            try:
                return joblib.load(path)
            except Exception:
                return None
    return None


model = load_artifact("best_model.joblib")
prep = load_artifact("preprocessor.joblib")  # exporté depuis la barre latérale de l'app Streamlit


class URLFeatures(BaseModel):
    URLLength: float
    DomainLength: float
    IsDomainIP: float
    CharContinuationRate: float
    TLDLegitimateProb: float
    URLCharProb: float
    TLDLength: float
    NoOfSubDomain: float
    HasObfuscation: float
    ObfuscationRatio: float
    LetterRatioInURL: float
    DegitRatioInURL: float
    SpacialCharRatioInURL: float
    NoOfURLRedirect: float
    NoOfSelfRedirect: float
    TLD: str = "com"


@app.get("/")
def root():
    return {
        "status": "online",
        "message": "API de Détection de Phishing opérationnelle.",
        "model_loaded": model is not None,
        "preprocessor_loaded": prep is not None,
        "documentation": "/docs",
    }


@app.post("/predict")
def predict(data: URLFeatures):
    if model is None or prep is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle ou préprocesseur non chargé (models/best_model.joblib et models/preprocessor.joblib requis).",
        )

    frame = pd.DataFrame([{f: getattr(data, f) for f in FEATURES}])
    tld = data.TLD.strip().lower().lstrip(".")
    frame["TLD_grouped"] = tld if tld in prep["top_tlds"] else "Other"

    numeric = prep["scaler"].transform(frame[FEATURES].fillna(prep["median"]))
    tld_encoded = prep["encoder"].transform(frame[["TLD_grouped"]])
    X = sp.hstack((numeric, tld_encoded), format="csr")

    # Dans PhiUSIIL : label 0 = phishing, label 1 = légitime
    proba_phishing = float(model.predict_proba(X)[0][list(model.classes_).index(0)])
    prediction = int(proba_phishing >= 0.50)

    if proba_phishing >= 0.70:
        risk_level = "Élevé 🔴"
    elif proba_phishing >= 0.35:
        risk_level = "Moyen 🟠"
    else:
        risk_level = "Faible 🟢"

    return {
        "prediction": prediction,  # 1 = phishing, 0 = légitime
        "label": "Phishing" if prediction == 1 else "Légitime",
        "proba_phishing": round(proba_phishing, 4),
        "proba_legitime": round(1.0 - proba_phishing, 4),
        "niveau_risque": risk_level,
        "confiance_pourcentage": round(max(proba_phishing, 1.0 - proba_phishing) * 100, 2),
    }