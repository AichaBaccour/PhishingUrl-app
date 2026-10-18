# =====================================================================
# 🛡️ PhiUSIIL — Système de détection de phishing (Streamlit)
# ---------------------------------------------------------------------
# Convention : dans ce dataset label = 1 -> URL légitime, label = 0 -> phishing.
# Dans toute l'application, la classe "positive" est le PHISHING
# (probabilité de phishing = predict_proba[:, 0]).
# =====================================================================
import io
import re
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp
import seaborn as sns
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from ucimlrepo import fetch_ucirepo

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except ImportError:  # xgboost absent : les autres modèles restent utilisables
    xgb, XGBClassifier = None, None

warnings.filterwarnings("ignore")

st.set_page_config(page_title="PhiUSIIL Phishing Detection", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------------------
# CONSTANTES
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
DEFAULT_API_URL = "https://phishingurl-app.onrender.com"
RANDOM_STATE = 42
SAVED_NAME = "XGBoost (modèle sauvegardé)"

# Les 15 variables numériques (même ordre que dans le notebook)
FEATURES = [
    "URLLength", "DomainLength", "IsDomainIP", "CharContinuationRate",
    "TLDLegitimateProb", "URLCharProb", "TLDLength", "NoOfSubDomain",
    "HasObfuscation", "ObfuscationRatio", "LetterRatioInURL", "DegitRatioInURL",
    "SpacialCharRatioInURL", "NoOfURLRedirect", "NoOfSelfRedirect",
]
BINARY_FEATURES = {"IsDomainIP", "HasObfuscation"}
RATIO_FEATURES = {
    "CharContinuationRate", "TLDLegitimateProb", "URLCharProb", "ObfuscationRatio",
    "LetterRatioInURL", "DegitRatioInURL", "SpacialCharRatioInURL",
}
NAMES_GROUPED = FEATURES + ["TLD"]  # les colonnes one-hot du TLD sont regroupées pour SHAP
COLORS = {"Phishing": "#e74c3c", "Légitime": "#2ecc71"}
MODEL_CHOICES = ["Logistic Regression", "Decision Tree", "Random Forest"]
if XGBClassifier is not None:
    MODEL_CHOICES.append("XGBoost (réentraîné)")


# ---------------------------------------------------------------------
# CHARGEMENT DU DATASET + PRÉTRAITEMENT (identique au notebook)
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Téléchargement et préparation du dataset PhiUSIIL (première fois : environ 1 minute)...")
def load_pipeline():
    ds = fetch_ucirepo(id=967)
    X = ds.data.features
    y = ds.data.targets.iloc[:, 0].astype(int)

    missing = [f for f in FEATURES if f not in X.columns]
    if missing:
        raise ValueError(f"Colonnes introuvables dans le dataset : {missing}")

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[["URL", "TLD"] + num_cols].copy()
    n_missing = int(X.isna().sum().sum())
    n_dups = int(X.duplicated().sum())

    # Même découpage que le notebook : 80/20, stratifié, random_state=42
    x_tr, x_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    x_tr, x_te = x_tr.copy(), x_te.copy()

    top30 = x_tr["TLD"].value_counts().head(30).index.tolist()
    for part in (x_tr, x_te):
        part["TLD_grouped"] = part["TLD"].where(part["TLD"].isin(top30), "Other")

    med = x_tr[FEATURES].median()
    scaler = StandardScaler().fit(x_tr[FEATURES].fillna(med))
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True).fit(x_tr[["TLD_grouped"]])

    def build(frame):
        num = scaler.transform(frame[FEATURES].fillna(med))
        return sp.hstack((num, enc.transform(frame[["TLD_grouped"]])), format="csr")

    X_train, X_test = build(x_tr), build(x_te)
    tld_names = [c.replace("TLD_grouped_", "TLD=") for c in enc.get_feature_names_out()]

    data = X.assign(label=y)
    data["classe"] = np.where(data["label"] == 0, "Phishing", "Légitime")

    saved_model, saved_auc = None, None
    if MODEL_PATH.exists():
        try:
            m = joblib.load(MODEL_PATH)
            if getattr(m, "n_features_in_", None) == X_test.shape[1]:
                saved_model = m
                col = list(m.classes_).index(0)
                saved_auc = float(roc_auc_score((y_te.values == 0).astype(int), m.predict_proba(X_test)[:, col]))
        except Exception:
            saved_model = None

    return SimpleNamespace(
        data=data, sample=data.sample(min(20000, len(data)), random_state=RANDOM_STATE),
        x_train=x_tr, x_test=x_te, y_train=y_tr.values, y_test=y_te.values,
        X_train=X_train, X_test=X_test,
        feat_names=FEATURES + tld_names, top30=top30, med=med, scaler=scaler, enc=enc,
        build=build, n_missing=n_missing, n_dups=n_dups,
        corr15=data[FEATURES].corr(), corr_all=data[num_cols].corr(),
        tld_prob=y_tr.groupby(x_tr["TLD"]).mean(), global_legit=float(y_tr.mean()),
        saved_model=saved_model, saved_auc=saved_auc,
    )


try:
    P = load_pipeline()
except Exception as exc:
    st.error(f"❌ Impossible de charger le dataset PhiUSIIL : {exc}")
    st.stop()

Y_TEST = (P.y_test == 0).astype(int)  # 1 = phishing

# ---------------------------------------------------------------------
# ÉTAT DE SESSION
# ---------------------------------------------------------------------
if "models" not in st.session_state:
    st.session_state.models = {}
    st.session_state.fit_times = {}
    st.session_state.probs = {}
    st.session_state.shap_cache = {}
    if P.saved_model is not None:
        st.session_state.models[SAVED_NAME] = P.saved_model


def phish_col(model):
    return list(model.classes_).index(0)


def phish_proba(name):
    """Probabilité de phishing sur tout le jeu de test (mise en cache)."""
    cache = st.session_state.probs
    if name not in cache:
        model = st.session_state.models[name]
        cache[name] = model.predict_proba(P.X_test)[:, phish_col(model)]
    return cache[name]


def metrics_at(y, p, t=0.5):
    pred = (p >= t).astype(int)
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y, p),
    }


def risk_level(p):
    if p >= 0.70:
        return "Élevé 🔴"
    if p >= 0.35:
        return "Moyen 🟠"
    return "Faible 🟢"


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def need_models():
    if not st.session_state.models:
        st.info("Aucun modèle disponible. Va d'abord dans l'onglet « 🤖 Laboratoire de modèles » pour en entraîner un.")
        return False
    return True


# ---------------------------------------------------------------------
# SHAP (valeurs exprimées dans le sens « phishing » : positif = plus de risque)
# ---------------------------------------------------------------------
def group_tld(vals):
    return np.column_stack([vals[:, : len(FEATURES)], vals[:, len(FEATURES):].sum(axis=1)])


def compute_contributions(model, X_sparse):
    """Retourne (contributions groupées [n, 16], valeur de base [n], unité)."""
    if XGBClassifier is not None and isinstance(model, XGBClassifier):
        contrib = model.get_booster().predict(xgb.DMatrix(X_sparse), pred_contribs=True)
        return group_tld(-contrib[:, :-1]), -contrib[:, -1], "log-odds"

    import shap

    dense = X_sparse.toarray()
    if isinstance(model, LogisticRegression):
        background = P.X_train[:200].toarray()
        explainer = shap.LinearExplainer(model, background)
        sv = explainer.shap_values(dense)
        base = float(np.ravel(explainer.expected_value)[0])
        return group_tld(-np.asarray(sv)), np.full(len(dense), -base), "log-odds"

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(dense)
    if isinstance(sv, list):
        sv = sv[1]
    elif np.ndim(sv) == 3:
        sv = sv[:, :, 1]
    base = float(np.ravel(explainer.expected_value)[-1])
    return group_tld(-np.asarray(sv)), np.full(len(dense), 1 - base), "probabilité"


def fmt(v):
    return v if isinstance(v, str) else f"{float(v):.3g}"


def plot_local(contrib, display_values, unit, top=10):
    order = np.argsort(np.abs(contrib))[::-1][:top][::-1]
    labels = [f"{NAMES_GROUPED[i]} = {fmt(display_values[i])}" for i in order]
    colors = [COLORS["Phishing"] if contrib[i] > 0 else COLORS["Légitime"] for i in order]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labels, contrib[order], color=colors)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel(f"Contribution vers PHISHING ({unit}) — rouge : augmente le risque, vert : le réduit")
    plt.tight_layout()
    return fig


def plain_language(contrib, display_values, top=3):
    lines = []
    for i in np.argsort(np.abs(contrib))[::-1][:top]:
        sens = "augmente" if contrib[i] > 0 else "réduit"
        lines.append(f"- **{NAMES_GROUPED[i]}** = {fmt(display_values[i])} → {sens} le risque de phishing.")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# EXTRACTION DE VARIABLES À PARTIR D'UNE URL (estimation)
# ---------------------------------------------------------------------
def url_features(url):
    u = url.strip()
    parsed = urlparse(u if "://" in u else "http://" + u)
    host = (parsed.hostname or "").lower()
    is_ip = int(bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host)))
    parts = host.split(".")
    tld = "" if is_ip or not parts else parts[-1]
    n = max(len(u), 1)
    letters = sum(c.isalpha() for c in u)
    digits = sum(c.isdigit() for c in u)
    n_obf = len(re.findall(r"%[0-9a-fA-F]{2}", u))
    vals = {
        "URLLength": float(len(u)),
        "DomainLength": float(len(host)),
        "IsDomainIP": float(is_ip),
        "TLDLength": float(len(tld)),
        "NoOfSubDomain": float(max(len(parts) - 2, 0)) if not is_ip else 0.0,
        "HasObfuscation": float(n_obf > 0),
        "ObfuscationRatio": min(3 * n_obf / n, 1.0),
        "LetterRatioInURL": letters / n,
        "DegitRatioInURL": digits / n,
        "SpacialCharRatioInURL": (n - letters - digits) / n,
    }
    return vals, tld


def prefill_from_url():
    url = st.session_state.get("url_text", "")
    if not url.strip():
        return
    vals, tld = url_features(url)
    for k, v in vals.items():
        st.session_state[f"in_{k}"] = v
    st.session_state["in_TLD"] = tld if tld in P.top30 else "Other"
    st.session_state["in_TLDLegitimateProb"] = float(P.tld_prob.get(tld, P.global_legit))


# ---------------------------------------------------------------------
# ENTRAÎNEMENT
# ---------------------------------------------------------------------
def make_model(name):
    if name == "Logistic Regression":
        return LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=15, random_state=RANDOM_STATE)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1)
    return XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE,
                         eval_metric="logloss", n_jobs=-1)


def train_models(selected, n_rows):
    idx = np.random.RandomState(RANDOM_STATE).choice(P.X_train.shape[0], size=min(n_rows, P.X_train.shape[0]), replace=False)
    Xs, ys = P.X_train[idx], P.y_train[idx]
    bar = st.progress(0.0, text="Entraînement...")
    for i, name in enumerate(selected):
        bar.progress(i / len(selected), text=f"Entraînement : {name}...")
        t0 = time.time()
        model = make_model(name).fit(Xs, ys)
        st.session_state.models[name] = model
        st.session_state.fit_times[name] = round(time.time() - t0, 2)
        st.session_state.probs.pop(name, None)
        st.session_state.shap_cache.pop(name, None)
    bar.progress(1.0, text="Entraînement terminé ✅")


# =====================================================================
# EN-TÊTE + BARRE LATÉRALE
# =====================================================================
st.title("🛡️ Système de Détection de Phishing — Dataset PhiUSIIL")
st.caption("Exploration, entraînement de modèles, seuil de décision, explicabilité SHAP, test d'URL et analyse des erreurs. "
           "Classe positive = **phishing**.")

if P.saved_model is None:
    st.warning("Le modèle `models/best_model.joblib` est absent ou incompatible (46 colonnes attendues). "
               "Entraîne un modèle dans l'onglet « Laboratoire de modèles ».")
elif P.saved_auc is not None and P.saved_auc < 0.95:
    st.warning(f"Le modèle sauvegardé obtient seulement un ROC-AUC de {P.saved_auc:.3f} sur le jeu de test : "
               "il ne correspond sans doute pas à ce prétraitement.")

st.sidebar.header("⚙️ Paramètres")
train_rows = st.sidebar.slider("Taille de l'échantillon d'entraînement (Laboratoire)", 10_000, int(P.X_train.shape[0]), 60_000, 10_000,
                               help="Plus c'est grand, plus c'est précis mais plus c'est long.")
st.sidebar.divider()
st.sidebar.subheader("🔌 API FastAPI (optionnel)")
api_url = st.sidebar.text_input("URL de l'API", DEFAULT_API_URL).rstrip("/")
if st.sidebar.button("Tester la connexion"):
    try:
        r = requests.get(f"{api_url}/", timeout=60)
        if r.status_code == 200:
            st.sidebar.success("🟢 API connectée")
        else:
            st.sidebar.error(f"❌ Code HTTP {r.status_code}")
    except Exception:
        st.sidebar.error("❌ API injoignable (serveur en veille ou URL incorrecte)")
st.sidebar.divider()
buffer = io.BytesIO()
joblib.dump({"scaler": P.scaler, "encoder": P.enc, "median": P.med, "top_tlds": P.top30, "features": FEATURES}, buffer)
st.sidebar.download_button("📦 Exporter preprocessor.joblib (pour l'API)", buffer.getvalue(), "preprocessor.joblib",
                           help="À placer dans le dossier models/ du dépôt, à côté de best_model.joblib.")

tabs = st.tabs([
    "📋 Aperçu", "🔎 Exploration", "🤖 Laboratoire de modèles", "🎚️ Seuil de décision",
    "🧠 Explicabilité (SHAP)", "🎯 Tester une URL", "⚠️ Analyse des erreurs",
])

# =====================================================================
# 1. APERÇU
# =====================================================================
with tabs[0]:
    st.header("Aperçu du jeu de données")
    n_total = len(P.data)
    n_phish = int((P.data["label"] == 0).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("URLs au total", f"{n_total:,}")
    c2.metric("Phishing (0)", f"{n_phish:,} ({n_phish / n_total:.1%})")
    c3.metric("Légitimes (1)", f"{n_total - n_phish:,} ({(n_total - n_phish) / n_total:.1%})")
    c4.metric("Variables utilisées", f"{len(FEATURES)} + TLD")

    st.subheader("10 premières lignes")
    st.dataframe(P.data[["URL", "TLD"] + FEATURES + ["classe"]].head(10))

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.pie([n_phish, n_total - n_phish], labels=["Phishing", "Légitime"], autopct="%1.1f%%",
               colors=[COLORS["Phishing"], COLORS["Légitime"]], startangle=90)
        show(fig)
    with right:
        st.subheader("Contrôle qualité")
        st.success(f"Valeurs manquantes : {P.n_missing}")
        st.success(f"Lignes en double : {P.n_dups}")
        st.info(f"Découpage : {len(P.y_train):,} URLs d'entraînement / {len(P.y_test):,} de test (80/20, stratifié, graine 42).")

    with st.expander("Pourquoi seulement 15 variables ?"):
        st.write(
            "Les variables issues du contenu de la page (titre, HTML, formulaires, HTTPS, nombre d'images, etc.) "
            "ont été retirées parce qu'elles trahissent presque directement la classe : elles donneraient des scores "
            "excellents mais irréalistes pour une détection à partir de l'URL seule. Les comptages bruts "
            "(nombre de lettres, de chiffres, de « = », etc.) ont aussi été retirés au profit des ratios, plus comparables "
            "d'une URL à l'autre. Le TLD est ajouté en one-hot sur les 30 plus fréquents (les autres sont regroupés dans « Other »), "
            "ce qui donne 15 + 31 = 46 colonnes en entrée du modèle."
        )

    st.download_button("⬇️ Télécharger un échantillon (1 000 lignes, CSV)",
                       P.data[["URL", "TLD"] + FEATURES + ["classe"]].head(1000).to_csv(index=False).encode("utf-8"),
                       "echantillon_phiusiil.csv", "text/csv")

# =====================================================================
# 2. EXPLORATION
# =====================================================================
with tabs[1]:
    st.header("Exploration des variables")
    feat = st.selectbox("Variable à analyser", FEATURES, key="eda_feat")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.histplot(data=P.sample, x=feat, hue="classe", palette=COLORS, bins=40, ax=ax)
        ax.set_title("Distribution par classe (échantillon de 20 000 URLs)")
        show(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.boxplot(data=P.sample, x="classe", y=feat, hue="classe", palette=COLORS, legend=False, ax=ax)
        show(fig)

    st.subheader("Corrélation entre les 15 variables")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.heatmap(P.corr15, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    show(fig)

    with st.expander("Corrélation de toutes les variables numériques du dataset"):
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.heatmap(P.corr_all, cmap="coolwarm", ax=ax)
        show(fig)

    st.subheader("Nuage de points")
    s1, s2 = st.columns(2)
    fx = s1.selectbox("Axe X", FEATURES, index=0, key="sc_x")
    fy = s2.selectbox("Axe Y", FEATURES, index=5, key="sc_y")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=P.sample.head(5000), x=fx, y=fy, hue="classe", palette=COLORS, alpha=0.5, s=12, ax=ax)
    show(fig)

# =====================================================================
# 3. LABORATOIRE DE MODÈLES
# =====================================================================
with tabs[2]:
    st.header("Entraîner et comparer des modèles")
    st.caption("Les modèles sont entraînés avec le même prétraitement que le notebook et évalués sur le même jeu de test (phishing = classe positive).")
    chosen = st.multiselect("Modèles à entraîner", MODEL_CHOICES, default=[m for m in MODEL_CHOICES if m != "Random Forest"])
    if st.button("🚀 Entraîner", disabled=not chosen):
        train_models(chosen, train_rows)

    if need_models():
        names = list(st.session_state.models)

        st.subheader("Performances sur le jeu de test (seuil 0,50)")
        rows = []
        for n in names:
            m = metrics_at(Y_TEST, phish_proba(n))
            m["Modèle"] = n
            m["Temps fit (s)"] = st.session_state.fit_times.get(n, np.nan)
            rows.append(m)
        table = pd.DataFrame(rows).set_index("Modèle").sort_values("F1", ascending=False)
        st.dataframe(table.style.format({c: "{:.4f}" for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]}, na_rep="—"))
        st.download_button("⬇️ Télécharger les métriques (CSV)", table.to_csv().encode("utf-8"), "metriques_modeles.csv", "text/csv")

        st.subheader("Courbes ROC et Précision-Rappel")
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
        for n in names:
            p = phish_proba(n)
            fpr, tpr, _ = roc_curve(Y_TEST, p)
            a1.plot(fpr, tpr, label=f"{n} (AUC={roc_auc_score(Y_TEST, p):.4f})")
            pr, rc, _ = precision_recall_curve(Y_TEST, p)
            a2.plot(rc, pr, label=n)
        a1.plot([0, 1], [0, 1], "k--", lw=0.8)
        a1.set(xlabel="Taux de faux positifs", ylabel="Taux de vrais positifs", title="ROC")
        a2.set(xlabel="Rappel", ylabel="Précision", title="Précision-Rappel")
        a1.legend(fontsize=8)
        show(fig)

        st.subheader("Détails d'un modèle")
        sel = st.selectbox("Modèle", names, key="lab_model")
        d1, d2 = st.columns(2)
        with d1:
            cm = confusion_matrix(Y_TEST, (phish_proba(sel) >= 0.5).astype(int), labels=[0, 1])
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                        xticklabels=["Légitime", "Phishing"], yticklabels=["Légitime", "Phishing"])
            ax.set(xlabel="Prédit", ylabel="Réel", title="Matrice de confusion (seuil 0,50)")
            show(fig)
        with d2:
            mdl = st.session_state.models[sel]
            if hasattr(mdl, "feature_importances_"):
                imp = pd.Series(mdl.feature_importances_, index=P.feat_names)
            elif hasattr(mdl, "coef_"):
                imp = pd.Series(np.abs(mdl.coef_[0]), index=P.feat_names)
            else:
                imp = None
            if imp is not None:
                fig, ax = plt.subplots(figsize=(5, 3.8))
                imp.sort_values().tail(12).plot.barh(ax=ax, color="#3498db")
                ax.set_title("Importance des variables (top 12)")
                show(fig)

        pt, pp = calibration_curve(Y_TEST, phish_proba(sel), n_bins=10)
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        ax.plot(pp, pt, "o-", label=sel)
        ax.plot([0, 1], [0, 1], "k--", label="Calibration parfaite")
        ax.set(xlabel="Probabilité prédite de phishing", ylabel="Fréquence réelle", title="Calibration")
        ax.legend(fontsize=8)
        show(fig)

# =====================================================================
# 4. SEUIL DE DÉCISION
# =====================================================================
with tabs[3]:
    st.header("Réglage du seuil de décision (calcul réel)")
    if need_models():
        c1, c2 = st.columns(2)
        sel = c1.selectbox("Modèle", list(st.session_state.models), key="thr_model")
        t = c2.slider("Seuil de décision (probabilité de phishing)", 0.0, 1.0, 0.50, 0.01, key="thr_value")
        p = phish_proba(sel)
        pred = (p >= t).astype(int)
        cm = confusion_matrix(Y_TEST, pred, labels=[0, 1])
        m = metrics_at(Y_TEST, p, t)

        l, r = st.columns(2)
        with l:
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                        xticklabels=["Légitime", "Phishing"], yticklabels=["Légitime", "Phishing"])
            ax.set(xlabel="Prédit", ylabel="Réel")
            show(fig)
        with r:
            k1, k2 = st.columns(2)
            k1.metric("Precision", f"{m['Precision']:.4f}")
            k2.metric("Recall", f"{m['Recall']:.4f}")
            k1.metric("F1", f"{m['F1']:.4f}")
            k2.metric("Accuracy", f"{m['Accuracy']:.4f}")
            st.write(f"🚨 **{cm[1, 0]:,}** phishings laissés passer (faux négatifs) — "
                     f"🚫 **{cm[0, 1]:,}** sites légitimes bloqués (faux positifs).")

        ts = np.linspace(0, 1, 101)
        pm = p[None, :] >= ts[:, None]
        pos, neg = Y_TEST[None, :] == 1, Y_TEST[None, :] == 0
        tp = (pm & pos).sum(1).astype(float)
        fp = (pm & neg).sum(1).astype(float)
        fn = (~pm & pos).sum(1).astype(float)
        prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)

        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.plot(ts, prec, label="Precision")
        ax.plot(ts, rec, label="Recall")
        ax.plot(ts, f1, label="F1")
        ax.axvline(t, color="grey", ls="--", label="Seuil choisi")
        ax.set(xlabel="Seuil", ylabel="Valeur")
        ax.legend()
        show(fig)

        st.info(f"Seuil qui maximise le F1 : **{ts[np.argmax(f1)]:.2f}** (F1 = {f1.max():.4f}).")
        ok = ts[rec >= 0.99]
        if len(ok):
            st.info(f"Seuil le plus élevé qui garde un rappel ≥ 99 % : **{ok.max():.2f}**.")
        st.caption("Baisser le seuil attrape plus de phishings mais bloque plus de sites légitimes ; l'augmenter fait l'inverse.")

# =====================================================================
# 5. EXPLICABILITÉ (SHAP)
# =====================================================================
with tabs[4]:
    st.header("🧠 IA explicable")
    st.caption("Les contributions sont exprimées vers la classe **phishing** : positif (rouge) = augmente le risque, négatif (vert) = le réduit. "
               "Les colonnes one-hot du TLD sont regroupées en une seule variable « TLD ».")
    if need_models():
        sel = st.selectbox("Modèle à expliquer", list(st.session_state.models), key="xai_model")
        shap_idx = np.random.RandomState(1).choice(P.X_test.shape[0], size=500, replace=False)
        if st.button("Calculer les valeurs SHAP (500 URLs de test)"):
            try:
                with st.spinner("Calcul en cours..."):
                    contrib, base, unit = compute_contributions(st.session_state.models[sel], P.X_test[shap_idx])
                st.session_state.shap_cache[sel] = (contrib, base, unit)
            except Exception as exc:
                st.error(f"Calcul SHAP impossible pour ce modèle : {exc}")

        if sel in st.session_state.shap_cache:
            contrib, base, unit = st.session_state.shap_cache[sel]
            raw = P.x_test.iloc[shap_idx]

            st.subheader("Importance globale")
            imp = pd.Series(np.abs(contrib).mean(axis=0), index=NAMES_GROUPED).sort_values()
            fig, ax = plt.subplots(figsize=(6, 4.5))
            imp.plot.barh(ax=ax, color="#3498db")
            ax.set_xlabel(f"Contribution absolue moyenne ({unit})")
            show(fig)

            st.subheader("Effet d'une variable")
            f_sel = st.selectbox("Variable", FEATURES, key="xai_feat")
            j = FEATURES.index(f_sel)
            fig, ax = plt.subplots(figsize=(6, 3.8))
            colors = np.where(P.y_test[shap_idx] == 0, COLORS["Phishing"], COLORS["Légitime"])
            ax.scatter(raw[f_sel].values, contrib[:, j], c=colors, s=14, alpha=0.6)
            ax.axhline(0, color="grey", lw=0.8)
            ax.set(xlabel=f_sel, ylabel=f"Contribution vers phishing ({unit})", title="Rouge : phishing réel — Vert : légitime réel")
            show(fig)

            st.subheader("Explication d'une URL précise")
            pos = st.slider("URL de l'échantillon", 0, len(shap_idx) - 1, 0, key="xai_pos")
            row = raw.iloc[pos]
            real = "Phishing" if P.y_test[shap_idx[pos]] == 0 else "Légitime"
            st.write(f"`{row['URL']}`")
            st.write(f"Classe réelle : **{real}** — probabilité de phishing prédite : **{phish_proba(sel)[shap_idx[pos]]:.2%}**")
            display = [row[f] for f in FEATURES] + [row["TLD_grouped"]]
            show(plot_local(contrib[pos], display, unit))
            st.markdown(plain_language(contrib[pos], display))

# =====================================================================
# 6. TESTER UNE URL
# =====================================================================
with tabs[5]:
    st.header("🎯 Tester une URL")
    if need_models():
        for f in FEATURES:
            st.session_state.setdefault(f"in_{f}", float(P.med[f]))
        st.session_state.setdefault("in_TLD", "com" if "com" in P.top30 else P.top30[0])

        st.text_input("Colle une URL complète pour pré-remplir les variables", key="url_text",
                      placeholder="https://www.exemple.com/chemin?x=1")
        st.button("Pré-remplir depuis l'URL", on_click=prefill_from_url)
        st.caption("Estimation à partir du texte de l'URL : les valeurs exactes du dataset peuvent différer légèrement. "
                   "Les variables de redirection et les probabilités propres au dataset gardent leur valeur médiane (modifiables ci-dessous).")

        cols = st.columns(3)
        for i, f in enumerate(FEATURES):
            with cols[i % 3]:
                if f in BINARY_FEATURES:
                    st.selectbox(f, [0.0, 1.0], key=f"in_{f}")
                elif f in RATIO_FEATURES:
                    st.number_input(f, min_value=0.0, max_value=1.0, step=0.01, format="%.4f", key=f"in_{f}")
                else:
                    st.number_input(f, min_value=0.0, step=1.0, format="%.0f", key=f"in_{f}")
        st.selectbox("TLD", P.top30 + ["Other"], key="in_TLD")

        c1, c2 = st.columns(2)
        model_name = c1.selectbox("Modèle", list(st.session_state.models), key="pred_model")
        threshold = c2.slider("Seuil de décision", 0.0, 1.0, 0.50, 0.01, key="pred_thr")
        compare_api = st.checkbox("Comparer avec l'API FastAPI")

        if st.button("🔍 Analyser"):
            values = {f: float(st.session_state[f"in_{f}"]) for f in FEATURES}
            frame = pd.DataFrame([values])
            frame["TLD_grouped"] = st.session_state["in_TLD"]
            Xr = P.build(frame)
            model = st.session_state.models[model_name]
            p = float(model.predict_proba(Xr)[0, phish_col(model)])

            st.markdown("---")
            if p >= threshold:
                st.error(f"🚨 **URL détectée comme PHISHING** (probabilité {p:.2%})")
            else:
                st.success(f"✅ **URL classée LÉGITIME** (probabilité de phishing {p:.2%})")
            st.progress(min(max(p, 0.0), 1.0))
            st.write(f"Niveau de risque : **{risk_level(p)}**")

            try:
                contrib, _, unit = compute_contributions(model, Xr)
                display = [values[f] for f in FEATURES] + [st.session_state["in_TLD"]]
                show(plot_local(contrib[0], display, unit))
                st.markdown(plain_language(contrib[0], display))
            except Exception as exc:
                st.info(f"Explication indisponible pour ce modèle : {exc}")

            if compare_api:
                try:
                    with st.spinner("Interrogation de l'API (peut prendre 30 s si le serveur dort)..."):
                        res = requests.post(f"{api_url}/predict", json={**values, "TLD": st.session_state["in_TLD"]}, timeout=60)
                    if res.status_code == 200:
                        data = res.json()
                        st.info(f"API : **{data.get('label')}** — probabilité de phishing {data.get('proba_phishing', 0):.2%}")
                    else:
                        st.warning(f"L'API a répondu {res.status_code}. Elle doit être mise à jour avec la nouvelle version de main.py.")
                except Exception as exc:
                    st.warning(f"API injoignable : {exc}")

# =====================================================================
# 7. ANALYSE DES ERREURS
# =====================================================================
with tabs[6]:
    st.header("⚠️ Analyse des erreurs")
    if need_models():
        c1, c2 = st.columns(2)
        sel = c1.selectbox("Modèle", list(st.session_state.models), key="err_model")
        t = c2.slider("Seuil", 0.0, 1.0, 0.50, 0.01, key="err_thr")
        p = phish_proba(sel)
        te = P.x_test[["URL", "TLD_grouped"] + FEATURES].copy()
        te["classe_reelle"] = np.where(Y_TEST == 1, "Phishing", "Légitime")
        te["proba_phishing"] = p
        te["erreur"] = np.where((p >= t) & (Y_TEST == 0), "Faux positif",
                         np.where((p < t) & (Y_TEST == 1), "Faux négatif", ""))
        errors = te[te["erreur"] != ""]

        k1, k2, k3 = st.columns(3)
        k1.metric("Erreurs au total", f"{len(errors):,} / {len(te):,}")
        k2.metric("🚨 Phishings manqués (FN)", f"{(errors['erreur'] == 'Faux négatif').sum():,}")
        k3.metric("🚫 Légitimes bloqués (FP)", f"{(errors['erreur'] == 'Faux positif').sum():,}")

        cols_show = ["URL", "TLD_grouped", "proba_phishing"]
        a, b = st.columns(2)
        with a:
            st.subheader("Phishings les moins bien détectés")
            st.dataframe(errors[errors["erreur"] == "Faux négatif"].sort_values("proba_phishing")[cols_show].head(20))
        with b:
            st.subheader("Légitimes les plus suspectés à tort")
            st.dataframe(errors[errors["erreur"] == "Faux positif"].sort_values("proba_phishing", ascending=False)[cols_show].head(20))

        st.subheader("Taux d'erreur par TLD")
        by_tld = te.assign(err=(te["erreur"] != "").astype(int)).groupby("TLD_grouped").agg(
            URLs=("err", "size"), taux_erreur=("err", "mean")).query("URLs >= 100").sort_values("taux_erreur", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        by_tld["taux_erreur"].sort_values().plot.barh(ax=ax, color="#e67e22")
        ax.set_xlabel("Taux d'erreur")
        show(fig)

        st.download_button("⬇️ Télécharger toutes les erreurs (CSV)", errors.to_csv(index=False).encode("utf-8"),
                           "erreurs_modele.csv", "text/csv")

st.divider()
st.caption("PhiUSIIL Phishing Detection | Projet pédagogique — valider les performances avant tout usage réel.")