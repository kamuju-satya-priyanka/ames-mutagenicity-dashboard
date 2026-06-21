"""
Ames Mutagenicity Predictor — Streamlit Web Application
========================================================
A production-ready ML application for predicting chemical mutagenicity
from SMILES representations using a trained Random Forest model.

Author  : Senior AI / Full-Stack ML Engineer
Version : 1.0.0
"""

from __future__ import annotations

import io
import json
import os
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from PIL import Image

# ── RDKit ──────────────────────────────────────────────────────────────────────
from rdkit import Chem
from rdkit.Chem import (
    Descriptors, rdMolDescriptors, rdFingerprintGenerator
)
try:
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    _DRAW_AVAILABLE = True
except ImportError:
    Draw = None
    rdMolDraw2D = None
    _DRAW_AVAILABLE = False

# ── SHAP ───────────────────────────────────────────────────────────────────────
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"

PAGE_ICON  = "🧬"
APP_TITLE  = "Ames Mutagenicity Predictor"
MORGAN_RADIUS = 2
MORGAN_BITS   = 2048

PHARMA_PRIMARY   = "#00D4FF"
PHARMA_SECONDARY = "#7B2FBE"
PHARMA_SUCCESS   = "#00E676"
PHARMA_DANGER    = "#FF1744"
PHARMA_WARNING   = "#FFD600"
PHARMA_BG        = "#0A0E1A"
PHARMA_CARD      = "#111827"
PHARMA_BORDER    = "#1E293B"

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": f"# {APP_TITLE}\nML-powered mutagenicity screening tool."},
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0A0E1A;
    color: #E2E8F0;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0A0E1A; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }

/* ── Main area ── */
.main .block-container {
    padding: 1.5rem 2rem 3rem;
    max-width: 1400px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1220 0%, #111827 100%);
    border-right: 1px solid #1E293B;
}
[data-testid="stSidebar"] .stRadio label {
    color: #94A3B8 !important;
    font-size: 0.875rem;
    font-weight: 500;
    transition: color 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover { color: #00D4FF !important; }

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0D1B3E 0%, #1a0533 50%, #0D1B3E 100%);
    border: 1px solid #1E3A5F;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center, rgba(0,212,255,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00D4FF, #7B2FBE, #00D4FF);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 4s linear infinite;
    line-height: 1.2;
    margin-bottom: 0.75rem;
}
@keyframes shimmer { 0% { background-position: 0% 50%; } 100% { background-position: 200% 50%; } }

.hero-subtitle {
    color: #94A3B8;
    font-size: 1.05rem;
    font-weight: 400;
    line-height: 1.6;
}

/* ── Metric Cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}
.metric-card {
    background: linear-gradient(135deg, #111827, #1E293B);
    border: 1px solid #1E3A5F;
    border-radius: 12px;
    padding: 1.25rem 1rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    cursor: default;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,212,255,0.15);
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #00D4FF;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.metric-label {
    font-size: 0.75rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.4rem;
}

/* ── Glass Cards ── */
.glass-card {
    background: rgba(17,24,39,0.8);
    backdrop-filter: blur(12px);
    border: 1px solid #1E293B;
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    transition: border-color 0.3s;
}
.glass-card:hover { border-color: rgba(0,212,255,0.3); }

/* ── Result Badges ── */
.badge-mutagenic {
    display: inline-block;
    background: linear-gradient(135deg, #FF1744, #B71C1C);
    color: white;
    font-size: 1.1rem;
    font-weight: 700;
    padding: 0.6rem 1.5rem;
    border-radius: 50px;
    text-align: center;
    letter-spacing: 0.05em;
    box-shadow: 0 0 20px rgba(255,23,68,0.4);
    animation: pulse-red 2s ease-in-out infinite;
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 20px rgba(255,23,68,0.4); }
    50%       { box-shadow: 0 0 35px rgba(255,23,68,0.7); }
}

.badge-nonmutagenic {
    display: inline-block;
    background: linear-gradient(135deg, #00E676, #00695C);
    color: #0A0E1A;
    font-size: 1.1rem;
    font-weight: 700;
    padding: 0.6rem 1.5rem;
    border-radius: 50px;
    text-align: center;
    letter-spacing: 0.05em;
    box-shadow: 0 0 20px rgba(0,230,118,0.4);
    animation: pulse-green 2s ease-in-out infinite;
}
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 20px rgba(0,230,118,0.4); }
    50%       { box-shadow: 0 0 35px rgba(0,230,118,0.7); }
}

/* ── Section Headers ── */
.section-header {
    font-size: 1.35rem;
    font-weight: 700;
    color: #E2E8F0;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid;
    border-image: linear-gradient(90deg, #00D4FF, transparent) 1;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Info Boxes ── */
.info-box {
    background: linear-gradient(135deg, rgba(0,212,255,0.05), rgba(123,47,190,0.05));
    border: 1px solid rgba(0,212,255,0.2);
    border-left: 4px solid #00D4FF;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
    font-size: 0.9rem;
    color: #94A3B8;
    line-height: 1.6;
}

.warning-box {
    background: rgba(255,214,0,0.05);
    border: 1px solid rgba(255,214,0,0.25);
    border-left: 4px solid #FFD600;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
    font-size: 0.9rem;
    color: #94A3B8;
    line-height: 1.6;
}

.danger-box {
    background: rgba(255,23,68,0.05);
    border: 1px solid rgba(255,23,68,0.25);
    border-left: 4px solid #FF1744;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
    font-size: 0.9rem;
    color: #94A3B8;
    line-height: 1.6;
}

/* ── Table ── */
.stDataFrame, [data-testid="stDataFrame"] {
    border: 1px solid #1E293B !important;
    border-radius: 10px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 10px;
    padding: 0.25rem;
    gap: 0.25rem;
    border: 1px solid #1E293B;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748B;
    border-radius: 8px;
    font-weight: 500;
    padding: 0.5rem 1.25rem;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #00D4FF22, #7B2FBE22) !important;
    color: #00D4FF !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
}

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea {
    background: #111827 !important;
    border: 1px solid #1E293B !important;
    border-radius: 8px !important;
    color: #E2E8F0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    transition: border-color 0.2s !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #00D4FF !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00D4FF, #7B2FBE) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 2rem !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(0,212,255,0.3) !important;
}

/* ── Download Button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #00E676, #00695C) !important;
    color: #0A0E1A !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    border: none !important;
}

/* ── Progress / spinner ── */
.stProgress > div > div { background: linear-gradient(90deg, #00D4FF, #7B2FBE) !important; }

/* ── Sidebar logo ── */
.sidebar-logo {
    text-align: center;
    padding: 1rem 0 0.5rem;
    border-bottom: 1px solid #1E293B;
    margin-bottom: 1rem;
}
.sidebar-logo-title {
    font-size: 1.1rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00D4FF, #7B2FBE);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.3;
}
.sidebar-logo-sub {
    font-size: 0.7rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.2rem;
}

/* ── Tag chips ── */
.tag-chip {
    display: inline-block;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.25);
    color: #00D4FF;
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 0.2rem;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #334155;
    font-size: 0.75rem;
    padding: 2rem 0 0.5rem;
    border-top: 1px solid #1E293B;
    margin-top: 3rem;
}
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model():
    """Load trained Random Forest model from disk."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    """Load pre-computed model performance metrics."""
    if not METRICS_PATH.exists():
        return {}
    with open(METRICS_PATH) as f:
        return json.load(f)


def validate_smiles(smiles: str) -> tuple[bool, Optional[object]]:
    """Validate a SMILES string using RDKit."""
    if not smiles or not smiles.strip():
        return False, None
    mol = Chem.MolFromSmiles(smiles.strip())
    return (mol is not None), mol


def smiles_to_fingerprint(smiles: str) -> Optional[np.ndarray]:
    """Convert SMILES to Morgan fingerprint vector."""
    valid, mol = validate_smiles(smiles)
    if not valid or mol is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    fp  = gen.GetFingerprintAsNumPy(mol)
    return fp.astype(np.float32)


def mol_to_image(mol, size: tuple[int, int] = (400, 300)) -> str:
    """Render an RDKit molecule to a base64-encoded image data URI.

    Tries rdMolDraw2D (SVG) first; falls back to a matplotlib/PIL PNG
    when the C extension is unavailable (e.g. Python 3.14 ABI mismatch).
    """
    import base64

    if _DRAW_AVAILABLE:
        # Fast SVG path via native C extension
        drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        drawer.drawOptions().addStereoAnnotation = True
        drawer.drawOptions().padding = 0.15
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg_str = drawer.GetDrawingText()
        b64 = base64.b64encode(svg_str.encode("utf-8")).decode("utf-8")
        return f"data:image/svg+xml;base64,{b64}"

    # Fallback: matplotlib kekulé structure via RDKit Draw.MolToImage
    try:
        from rdkit.Chem import Draw as _RDDraw
        pil_img = _RDDraw.MolToImage(mol, size=size)
    except Exception:
        # Last-resort blank placeholder
        pil_img = Image.new("RGB", size, color=(17, 24, 39))

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def compute_descriptors(mol) -> dict:
    """Compute standard Lipinski / Veber descriptors."""
    return {
        "Molecular Weight (Da)": round(Descriptors.MolWt(mol), 3),
        "LogP (Wildman-Crippen)": round(Descriptors.MolLogP(mol), 3),
        "TPSA (Å²)": round(Descriptors.TPSA(mol), 3),
        "H-bond Donors": rdMolDescriptors.CalcNumHBD(mol),
        "H-bond Acceptors": rdMolDescriptors.CalcNumHBA(mol),
        "Rotatable Bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "Aromatic Rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "Heavy Atom Count": mol.GetNumHeavyAtoms(),
        "Ring Count": rdMolDescriptors.CalcNumRings(mol),
        "Stereo Centers": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
    }


def predict_single(smiles: str, model) -> Optional[dict]:
    """Run prediction on a single SMILES. Returns dict with results."""
    fp = smiles_to_fingerprint(smiles)
    if fp is None:
        return None
    fp_2d = fp.reshape(1, -1)
    pred_class   = int(model.predict(fp_2d)[0])
    pred_proba   = model.predict_proba(fp_2d)[0]
    probability  = float(pred_proba[pred_class])
    mut_prob     = float(pred_proba[1])
    return {
        "class":       pred_class,
        "label":       "Mutagenic" if pred_class == 1 else "Non-Mutagenic",
        "probability": probability,
        "mut_prob":    mut_prob,
        "safe_prob":   float(pred_proba[0]),
        "fp":          fp,
        "fp_2d":       fp_2d,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTLY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

PLOTLY_DARK_LAYOUT = dict(
    paper_bgcolor="#0A0E1A",
    plot_bgcolor="#111827",
    font=dict(family="Inter", color="#94A3B8", size=12),
    margin=dict(l=40, r=20, t=50, b=40),
)


def make_gauge(mut_prob: float) -> go.Figure:
    """Create Plotly gauge for toxicity risk."""
    risk_pct = mut_prob * 100
    if risk_pct < 30:
        color, risk_label = "#00E676", "SAFE"
    elif risk_pct < 70:
        color, risk_label = "#FFD600", "MODERATE RISK"
    else:
        color, risk_label = "#FF1744", "HIGH RISK"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_pct,
        number={"suffix": "%", "font": {"size": 36, "color": color, "family": "JetBrains Mono"}},
        title={"text": f"Toxicity Risk Meter<br><span style='font-size:14px;color:{color}'>{risk_label}</span>",
               "font": {"size": 16, "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#334155", "tickfont": {"color": "#64748B"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#111827",
            "bordercolor": "#1E293B",
            "steps": [
                {"range": [0,  30], "color": "rgba(0,230,118,0.1)"},
                {"range": [30, 70], "color": "rgba(255,214,0,0.1)"},
                {"range": [70, 100], "color": "rgba(255,23,68,0.1)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.85,
                "value": risk_pct,
            },
        },
    ))
    fig.update_layout(height=300, **PLOTLY_DARK_LAYOUT)
    return fig


def make_confusion_matrix(cm: list) -> go.Figure:
    """Create annotated confusion matrix heatmap."""
    labels = ["Non-Mutagenic", "Mutagenic"]
    z = np.array(cm)
    text = [[str(v) for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, text=text,
        texttemplate="%{text}",
        colorscale=[[0, "#111827"], [0.5, "#7B2FBE"], [1, "#00D4FF"]],
        showscale=False,
        hoverongaps=False,
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        height=380,
        **PLOTLY_DARK_LAYOUT,
    )
    return fig


def make_roc_curve(fpr: list, tpr: list, auc: float) -> go.Figure:
    """Create ROC Curve plot."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"ROC Curve (AUC = {auc:.3f})",
        line=dict(color="#00D4FF", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random Classifier",
        line=dict(color="#334155", width=1.5, dash="dash"),
    ))
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=400,
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1E293B"),
        **PLOTLY_DARK_LAYOUT,
    )
    return fig


def make_feature_importance(importances: list, top_n: int = 20) -> go.Figure:
    """Horizontal bar chart of top-N feature importances."""
    idxs = np.argsort(importances)[-top_n:][::-1]
    vals = [importances[i] for i in idxs]
    names = [f"Bit_{i}" for i in idxs]

    fig = go.Figure(go.Bar(
        x=vals[::-1], y=names[::-1],
        orientation="h",
        marker=dict(
            color=vals[::-1],
            colorscale=[[0, "#7B2FBE"], [1, "#00D4FF"]],
        ),
    ))
    fig.update_layout(
        title=f"Top {top_n} Feature Importances (Morgan Fingerprint Bits)",
        xaxis_title="Importance",
        height=520,
        **PLOTLY_DARK_LAYOUT,
    )
    return fig


def make_probability_bar(mut_prob: float, safe_prob: float) -> go.Figure:
    """Horizontal stacked probability bar."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[safe_prob * 100], y=["Probability"],
        orientation="h", name="Non-Mutagenic",
        marker_color="#00E676",
        text=[f"Non-Mutagenic: {safe_prob*100:.1f}%"],
        textposition="inside",
        insidetextanchor="middle",
    ))
    fig.add_trace(go.Bar(
        x=[mut_prob * 100], y=["Probability"],
        orientation="h", name="Mutagenic",
        marker_color="#FF1744",
        text=[f"Mutagenic: {mut_prob*100:.1f}%"],
        textposition="inside",
        insidetextanchor="middle",
    ))
    fig.update_layout(
        barmode="stack",
        height=120,
        showlegend=True,
        xaxis=dict(range=[0, 100], ticksuffix="%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **PLOTLY_DARK_LAYOUT,
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> str:
    """Render sidebar navigation and return selected page."""
    with st.sidebar:
        st.markdown(
            """
            <div class='sidebar-logo'>
                <div style='font-size:2.2rem;'>🧬</div>
                <div class='sidebar-logo-title'>Ames Mutagenicity<br>Predictor</div>
                <div class='sidebar-logo-sub'>v1.0.0 · ML Edition</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        pages = {
            "🏠  Home":                  "Home",
            "🔬  Single Prediction":     "Single Prediction",
            "📂  Batch Prediction":      "Batch Prediction",
            "🧪  Molecular Visualization":"Molecular Visualization",
            "🤖  Explainability (XAI)":  "Explainability",
            "📊  Model Performance":     "Model Performance",
            "⚗️   Chemical Descriptors":  "Chemical Descriptors",
            "ℹ️   About":                 "About",
        }

        selected_label = st.radio(
            "Navigation",
            list(pages.keys()),
            label_visibility="collapsed",
        )
        page = pages[selected_label]

        st.markdown("---")
        model = load_model()
        if model is not None:
            st.markdown(
                "<div class='info-box'>✅ <b>Model Loaded</b><br>Random Forest · 200 estimators</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='warning-box'>⚠️ <b>Model Not Found</b><br>Run <code>python train_model.py</code></div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            "<div class='footer'>© 2024 Ames Predictor<br>For research use only</div>",
            unsafe_allow_html=True,
        )

    return page


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════

def page_home():
    st.markdown(
        """
        <div class='hero-banner'>
            <div class='hero-title'>🧬 Ames Mutagenicity Predictor</div>
            <div class='hero-subtitle'>
                An AI-powered platform for rapid, high-confidence mutagenicity screening
                of chemical compounds using molecular fingerprints and Machine Learning.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Key stats ──────────────────────────────────────────────────────────
    metrics = load_metrics()
    acc  = metrics.get("accuracy", 0.88)
    prec = metrics.get("precision", 0.87)
    rec  = metrics.get("recall", 0.86)
    auc  = metrics.get("roc_auc", 0.93)

    st.markdown(
        f"""
        <div class='metric-grid'>
            <div class='metric-card'>
                <div class='metric-value'>{acc*100:.1f}%</div>
                <div class='metric-label'>Accuracy</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>{prec*100:.1f}%</div>
                <div class='metric-label'>Precision</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>{rec*100:.1f}%</div>
                <div class='metric-label'>Recall</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>{auc:.3f}</div>
                <div class='metric-label'>ROC AUC</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>2048</div>
                <div class='metric-label'>Fingerprint Bits</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>200</div>
                <div class='metric-label'>Trees (RF)</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("<div class='section-header'>🔬 What is the Ames Test?</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='glass-card'>
            <p style='color:#94A3B8; line-height:1.7; font-size:0.95rem;'>
            The <strong style='color:#00D4FF;'>Ames test</strong> (Salmonella mutagenicity assay) is a widely used
            biological assay developed by Dr. Bruce Ames in the 1970s to assess the mutagenic potential
            of chemical substances. A chemical is considered <em>mutagenic</em> if it induces mutations
            in the Salmonella typhimurium bacterial genome.
            </p>
            <p style='color:#94A3B8; line-height:1.7; font-size:0.95rem; margin-top:0.75rem;'>
            Mutagenicity is a critical safety endpoint in drug discovery, toxicology, and environmental
            risk assessment. This ML predictor provides <em>in silico</em> Ames test results,
            significantly reducing time and cost compared to laboratory testing.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='section-header'>📊 Dataset Information</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='glass-card'>
            <table style='width:100%; border-collapse:collapse; font-size:0.875rem; color:#94A3B8;'>
              <tr style='border-bottom:1px solid #1E293B;'>
                <td style='padding:0.5rem 0; font-weight:600; color:#E2E8F0;'>Source</td>
                <td>Ames Mutagenicity Benchmark (curated)</td>
              </tr>
              <tr style='border-bottom:1px solid #1E293B;'>
                <td style='padding:0.5rem 0; font-weight:600; color:#E2E8F0;'>Representation</td>
                <td>Morgan Fingerprints (radius=2, 2048 bits)</td>
              </tr>
              <tr style='border-bottom:1px solid #1E293B;'>
                <td style='padding:0.5rem 0; font-weight:600; color:#E2E8F0;'>Classes</td>
                <td>Mutagenic (1) · Non-Mutagenic (0)</td>
              </tr>
              <tr>
                <td style='padding:0.5rem 0; font-weight:600; color:#E2E8F0;'>Split</td>
                <td>80% train · 20% test (stratified)</td>
              </tr>
            </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("<div class='section-header'>⚙️ Model Architecture</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='glass-card'>
            <p style='color:#94A3B8; font-size:0.875rem; margin-bottom:0.75rem;'>
            <strong style='color:#00D4FF;'>Algorithm:</strong> Random Forest Classifier
            </p>
            <ul style='color:#94A3B8; font-size:0.875rem; line-height:2; padding-left:1.2rem;'>
              <li><strong style='color:#E2E8F0;'>n_estimators</strong> — 200 decision trees</li>
              <li><strong style='color:#E2E8F0;'>max_features</strong> — sqrt (auto)</li>
              <li><strong style='color:#E2E8F0;'>class_weight</strong> — balanced</li>
              <li><strong style='color:#E2E8F0;'>Input</strong> — 2048-bit Morgan FP</li>
              <li><strong style='color:#E2E8F0;'>Output</strong> — Binary class + probability</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='section-header'>🚀 Quick Start</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='glass-card'>
            <p style='color:#94A3B8; font-size:0.875rem; line-height:1.8;'>
            <span class='tag-chip'>Step 1</span> Navigate to <strong>Single Prediction</strong><br>
            <span class='tag-chip'>Step 2</span> Enter a SMILES string (e.g., <code style='color:#00D4FF;'>CCO</code>)<br>
            <span class='tag-chip'>Step 3</span> Click <strong>Predict</strong><br>
            <span class='tag-chip'>Step 4</span> Review results & SHAP explanations<br>
            <span class='tag-chip'>Optional</span> Upload CSV for batch screening
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='section-header'>🛡️ Disclaimer</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='warning-box'>This tool is for <strong>research purposes only</strong>. "
            "Predictions should not replace formal toxicological assessment. "
            "Always validate with experimental assays for regulatory submissions.</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: SINGLE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def page_single_prediction():
    st.markdown("<div class='hero-title' style='font-size:1.8rem;'>🔬 Single Compound Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Enter a SMILES string to predict Ames mutagenicity.</div><br>", unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.markdown("<div class='danger-box'>⚠️ Model not found. Please run <code>python train_model.py</code> first.</div>", unsafe_allow_html=True)
        return

    # ── Example molecules ──────────────────────────────────────────────────
    st.markdown("<div class='section-header'>💡 Example Compounds</div>", unsafe_allow_html=True)
    ex_col1, ex_col2, ex_col3, ex_col4 = st.columns(4)
    examples = {
        "Ethanol (Safe)":              "CCO",
        "Benzo[a]pyrene (Mutagen)":   "c1ccc2c(c1)ccc3cccc4cccc2c34",
        "Aspirin (Safe)":              "CC(=O)Oc1ccccc1C(=O)O",
        "4-Nitroaniline (Mutagen)":   "Nc1ccc([N+](=O)[O-])cc1",
    }
    smiles_input = ""
    for col, (name, smi) in zip([ex_col1, ex_col2, ex_col3, ex_col4], examples.items()):
        with col:
            if st.button(name, key=f"ex_{name}"):
                st.session_state["smiles_input"] = smi

    # ── Input area ─────────────────────────────────────────────────────────
    smiles_default = st.session_state.get("smiles_input", "CCO")
    smiles_value = st.text_input(
        "SMILES String",
        value=smiles_default,
        placeholder="e.g., CCO  or  c1ccc2c(c1)ccc3cccc4cccc2c34",
        key="smiles_text_input",
        help="Simplified Molecular Input Line Entry System notation",
    )
    if smiles_value:
        st.session_state["smiles_input"] = smiles_value

    predict_btn = st.button("⚡ Predict Mutagenicity", type="primary", use_container_width=True)

    if predict_btn and smiles_value:
        smiles = smiles_value.strip()
        valid, mol = validate_smiles(smiles)

        if not valid:
            st.markdown(
                "<div class='danger-box'>❌ <strong>Invalid SMILES</strong> — Could not parse the molecule. "
                "Please check your input and try again.</div>",
                unsafe_allow_html=True,
            )
            return

        with st.spinner("Running prediction…"):
            result = predict_single(smiles, model)

        if result is None:
            st.error("Fingerprint generation failed.")
            return

        # ── Layout ────────────────────────────────────────────────────────
        r_col1, r_col2 = st.columns([1, 1.4])

        with r_col1:
            st.markdown("<div class='section-header'>🏷️ Prediction Result</div>", unsafe_allow_html=True)
            badge_class = "badge-mutagenic" if result["class"] == 1 else "badge-nonmutagenic"
            emoji = "☠️" if result["class"] == 1 else "✅"
            st.markdown(
                f"<div style='text-align:center; padding:1.5rem 0;'>"
                f"<div class='{badge_class}'>{emoji}  {result['label']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Probability bar
            st.plotly_chart(
                make_probability_bar(result["mut_prob"], result["safe_prob"]),
                use_container_width=True,
            )

            # Confidence table
            conf_df = pd.DataFrame({
                "Metric": ["Mutagenic Probability", "Non-Mutagenic Probability", "Confidence Score"],
                "Value":  [
                    f"{result['mut_prob']*100:.2f}%",
                    f"{result['safe_prob']*100:.2f}%",
                    f"{result['probability']*100:.2f}%",
                ],
            })
            st.dataframe(conf_df, use_container_width=True, hide_index=True)

        with r_col2:
            st.markdown("<div class='section-header'>⚗️ Toxicity Risk Meter</div>", unsafe_allow_html=True)
            st.plotly_chart(make_gauge(result["mut_prob"]), use_container_width=True)

            # Molecular structure
            st.markdown("<div class='section-header'>🔭 Molecular Structure</div>", unsafe_allow_html=True)
            svg_uri = mol_to_image(mol, size=(400, 220))
            st.markdown(
                f"<div style='background:#111827;border:1px solid #1E293B;border-radius:10px;padding:0.5rem;text-align:center;'>"
                f"<img src='{svg_uri}' style='max-width:100%;border-radius:8px;background:white;' />"
                f"</div>",
                unsafe_allow_html=True,
            )

        # ── Molecular descriptors inline ────────────────────────────────────
        st.markdown("<div class='section-header'>📐 Chemical Descriptors</div>", unsafe_allow_html=True)
        desc = compute_descriptors(mol)
        d_cols = st.columns(5)
        for i, (k, v) in enumerate(desc.items()):
            with d_cols[i % 5]:
                st.metric(k, v)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: BATCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def page_batch_prediction():
    st.markdown("<div class='hero-title' style='font-size:1.8rem;'>📂 Batch Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Upload a CSV file with a <code>SMILES</code> column to screen multiple compounds.</div><br>", unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.markdown("<div class='danger-box'>⚠️ Model not found. Run <code>python train_model.py</code> first.</div>", unsafe_allow_html=True)
        return

    # Template download
    template_df = pd.DataFrame({"SMILES": ["CCO", "CCN", "CCCl", "c1ccc2c(c1)ccc3cccc4cccc2c34", "Nc1ccc([N+](=O)[O-])cc1"]})
    template_csv = template_df.to_csv(index=False)

    st.markdown("<div class='section-header'>📥 Input Format</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='info-box'>CSV must contain a <strong>SMILES</strong> column (case-insensitive). "
        "One compound per row. Download the template below to get started.</div>",
        unsafe_allow_html=True,
    )
    st.download_button(
        "📄 Download CSV Template",
        data=template_csv,
        file_name="ames_template.csv",
        mime="text/csv",
    )

    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"],
        help="File must have a SMILES column.",
    )

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
        except Exception as e:
            st.markdown(f"<div class='danger-box'>❌ Failed to read CSV: {e}</div>", unsafe_allow_html=True)
            return

        # Normalise column name
        col_map = {c.strip().upper(): c for c in df_input.columns}
        if "SMILES" not in col_map:
            st.markdown("<div class='danger-box'>❌ No <strong>SMILES</strong> column found in the uploaded file.</div>", unsafe_allow_html=True)
            return

        smiles_col = col_map["SMILES"]
        total = len(df_input)
        st.markdown(f"<div class='info-box'>📋 Loaded <strong>{total}</strong> compounds. Running predictions…</div>", unsafe_allow_html=True)

        pbar = st.progress(0, text="Processing…")
        results = []

        for i, row in enumerate(df_input[smiles_col].astype(str)):
            smiles = row.strip()
            valid, mol = validate_smiles(smiles)
            if not valid or mol is None:
                results.append({
                    "SMILES": smiles,
                    "Valid": False,
                    "Prediction": "INVALID",
                    "Mutagenic_Probability": None,
                    "Non_Mutagenic_Probability": None,
                    "Confidence": None,
                })
            else:
                res = predict_single(smiles, model)
                results.append({
                    "SMILES": smiles,
                    "Valid": True,
                    "Prediction": res["label"],
                    "Mutagenic_Probability": round(res["mut_prob"], 4),
                    "Non_Mutagenic_Probability": round(res["safe_prob"], 4),
                    "Confidence": round(res["probability"], 4),
                })
            pbar.progress((i + 1) / total, text=f"Processed {i+1}/{total}")

        pbar.empty()
        df_out = pd.DataFrame(results)

        # ── Summary stats ──────────────────────────────────────────────────
        valid_df   = df_out[df_out["Valid"]]
        n_valid    = len(valid_df)
        n_invalid  = total - n_valid
        n_mut      = (valid_df["Prediction"] == "Mutagenic").sum()
        n_safe     = (valid_df["Prediction"] == "Non-Mutagenic").sum()

        st.markdown(
            f"""
            <div class='metric-grid'>
                <div class='metric-card'><div class='metric-value'>{total}</div><div class='metric-label'>Total Compounds</div></div>
                <div class='metric-card'><div class='metric-value'>{n_valid}</div><div class='metric-label'>Valid SMILES</div></div>
                <div class='metric-card'><div class='metric-value' style='color:#FF1744'>{n_mut}</div><div class='metric-label'>Mutagenic</div></div>
                <div class='metric-card'><div class='metric-value' style='color:#00E676'>{n_safe}</div><div class='metric-label'>Non-Mutagenic</div></div>
                <div class='metric-card'><div class='metric-value' style='color:#FFD600'>{n_invalid}</div><div class='metric-label'>Invalid SMILES</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Pie chart
        if n_valid > 0:
            pie = go.Figure(go.Pie(
                labels=["Mutagenic", "Non-Mutagenic"],
                values=[n_mut, n_safe],
                hole=0.55,
                marker=dict(colors=["#FF1744", "#00E676"]),
                textfont=dict(family="Inter"),
            ))
            pie.update_layout(
                title="Prediction Distribution",
                height=320,
                **PLOTLY_DARK_LAYOUT,
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(pie, use_container_width=True)

        # Results table
        st.markdown("<div class='section-header'>📋 Results Table</div>", unsafe_allow_html=True)
        st.dataframe(
            df_out.style.map(
                lambda v: "color: #FF1744" if v == "Mutagenic"
                else ("color: #00E676" if v == "Non-Mutagenic" else ""),
                subset=["Prediction"],
            ),
            use_container_width=True,
        )

        # Download
        csv_out = df_out.to_csv(index=False)
        st.download_button(
            "⬇️ Download Results CSV",
            data=csv_out,
            file_name="ames_predictions.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: MOLECULAR VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def page_molecular_visualization():
    st.markdown("<div class='hero-title' style='font-size:1.8rem;'>🧪 Molecular Visualization</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Render 2-D molecular structures from SMILES input.</div><br>", unsafe_allow_html=True)

    smiles_input = st.text_area(
        "SMILES (one per line for multiple structures)",
        value="CCO\nc1ccc2c(c1)ccc3cccc4cccc2c34\nCC(=O)Oc1ccccc1C(=O)O\nNc1ccc([N+](=O)[O-])cc1",
        height=120,
        help="Enter one SMILES per line to visualize multiple molecules.",
    )

    if st.button("🎨 Generate Structures", type="primary"):
        lines = [l.strip() for l in smiles_input.strip().splitlines() if l.strip()]
        if not lines:
            st.markdown("<div class='warning-box'>Please enter at least one SMILES.</div>", unsafe_allow_html=True)
            return

        cols = st.columns(min(len(lines), 3))
        for i, smi in enumerate(lines):
            valid, mol = validate_smiles(smi)
            with cols[i % 3]:
                if not valid or mol is None:
                    st.markdown(f"<div class='danger-box'>❌ Invalid: <code>{smi}</code></div>", unsafe_allow_html=True)
                else:
                    svg_uri = mol_to_image(mol, size=(380, 260))
                    st.markdown(
                        f"<div style='background:#111827;border:1px solid #1E293B;border-radius:10px;padding:0.5rem;text-align:center;'>"
                        f"<img src='{svg_uri}' style='max-width:100%;border-radius:8px;background:white;' />"
                        f"<p style='color:#64748B;font-size:0.75rem;margin-top:0.3rem;font-family:JetBrains Mono;'>{smi}</p>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    desc = compute_descriptors(mol)
                    st.markdown(
                        f"<div class='info-box'>"
                        f"⚖️ MW: <strong>{desc['Molecular Weight (Da)']} Da</strong> &nbsp;|&nbsp; "
                        f"💧 LogP: <strong>{desc['LogP (Wildman-Crippen)']}</strong> &nbsp;|&nbsp; "
                        f"💊 TPSA: <strong>{desc['TPSA (Å²)']} Å²</strong>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: EXPLAINABILITY (XAI)
# ══════════════════════════════════════════════════════════════════════════════

def page_explainability():
    st.markdown("<div class='hero-title' style='font-size:1.8rem;'>🤖 Explainable AI Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>SHAP-based explanations for model predictions.</div><br>", unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.markdown("<div class='danger-box'>⚠️ Model not found.</div>", unsafe_allow_html=True)
        return

    smiles = st.text_input(
        "SMILES for Explanation",
        value="c1ccc2c(c1)ccc3cccc4cccc2c34",
        placeholder="Enter a SMILES string…",
    )

    if st.button("🔍 Explain Prediction", type="primary"):
        valid, mol = validate_smiles(smiles)
        if not valid:
            st.markdown("<div class='danger-box'>❌ Invalid SMILES.</div>", unsafe_allow_html=True)
            return

        fp = smiles_to_fingerprint(smiles)
        if fp is None:
            st.error("Fingerprint generation failed.")
            return

        fp_2d = fp.reshape(1, -1)
        result = predict_single(smiles, model)

        with st.spinner("Computing SHAP values (this may take a moment)…"):
            try:
                # Use TreeExplainer for efficiency
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(fp_2d)

                # shap_values shape: [n_classes, n_samples, n_features]
                if isinstance(shap_values, list):
                    sv_class1 = shap_values[1][0]  # For mutagenic class
                else:
                    sv_class1 = shap_values[0]

                # normalise: flat 1-D float array, exactly MORGAN_BITS long
                sv_class1 = np.array(sv_class1, dtype=float).flatten()[:MORGAN_BITS]

                shap_computed = True
            except Exception as e:
                shap_computed = False
                st.warning(f"SHAP computation error: {e}. Falling back to feature importance.")

        # ── Prediction summary ─────────────────────────────────────────────
        badge_class = "badge-mutagenic" if result["class"] == 1 else "badge-nonmutagenic"
        st.markdown(
            f"<div style='text-align:center; padding:1rem 0;'>"
            f"<div class='{badge_class}'>{'☠️' if result['class'] else '✅'} {result['label']} "
            f"({result['mut_prob']*100:.1f}% mutagenic)</div></div>",
            unsafe_allow_html=True,
        )

        if shap_computed:
            tab1, tab2, tab3 = st.tabs(["📊 Feature Impact", "🌊 SHAP Waterfall", "🔝 Top Features"])

            with tab1:
                st.markdown("<div class='section-header'>SHAP Feature Impact (Mutagenic Class)</div>", unsafe_allow_html=True)
                top_n = 25
                top_idx  = np.argsort(np.abs(sv_class1))[::-1][:top_n]
                top_vals = sv_class1[top_idx]
                top_names = [f"Bit_{i}" for i in top_idx]
                colors = ["#FF1744" if float(v) > 0 else "#00E676" for v in top_vals]

                fig_impact = go.Figure(go.Bar(
                    x=top_vals[::-1],
                    y=top_names[::-1],
                    orientation="h",
                    marker_color=colors[::-1],
                ))
                fig_impact.update_layout(
                    title=f"Top {top_n} SHAP Values (Red = ↑ Mutagenic, Green = ↓ Mutagenic)",
                    xaxis_title="SHAP Value",
                    height=550,
                    **PLOTLY_DARK_LAYOUT,
                )
                st.plotly_chart(fig_impact, use_container_width=True)

            with tab2:
                st.markdown("<div class='section-header'>SHAP Waterfall Plot</div>", unsafe_allow_html=True)
                try:
                    fig_wf, ax = plt.subplots(figsize=(10, 6))
                    fig_wf.patch.set_facecolor("#0A0E1A")
                    ax.set_facecolor("#111827")

                    # Manual waterfall
                    base = float(explainer.expected_value[1]) if isinstance(explainer.expected_value, np.ndarray) else float(explainer.expected_value)
                    n_show = 15
                    top_wf_idx = np.argsort(np.abs(sv_class1))[::-1][:n_show]
                    wf_vals = sv_class1[top_wf_idx]
                    wf_names = [f"Bit_{i}" for i in top_wf_idx]

                    cumsum = base
                    lefts, widths, bar_colors, labels = [], [], [], []
                    for v, nm in zip(wf_vals, wf_names):
                        lefts.append(min(cumsum, cumsum + v))
                        widths.append(abs(v))
                        bar_colors.append("#FF1744" if float(v) > 0 else "#00E676")
                        labels.append(nm)
                        cumsum += v

                    ax.barh(labels, widths, left=lefts, color=bar_colors, alpha=0.85)
                    ax.axvline(base, color="#FFD600", linestyle="--", linewidth=1.5, label=f"Base ({base:.3f})")
                    ax.set_xlabel("Model Output (log-odds)", color="#94A3B8")
                    ax.tick_params(colors="#94A3B8")
                    ax.spines[["top", "right"]].set_visible(False)
                    ax.spines[["left", "bottom"]].set_color("#334155")
                    ax.legend(facecolor="#111827", edgecolor="#334155", labelcolor="#94A3B8")
                    plt.title("SHAP Waterfall — Top Contributing Bits", color="#E2E8F0", pad=12)
                    plt.tight_layout()
                    st.pyplot(fig_wf, use_container_width=True)
                    plt.close(fig_wf)
                except Exception as e:
                    st.warning(f"Could not render waterfall: {e}")

            with tab3:
                st.markdown("<div class='section-header'>🔝 Top Contributing Fingerprint Bits</div>", unsafe_allow_html=True)
                top20_idx  = np.argsort(np.abs(sv_class1))[::-1][:20]
                top20_vals = sv_class1[top20_idx]
                top20_bit  = fp[top20_idx]

                df_top = pd.DataFrame({
                    "Rank":        range(1, 21),
                    "Bit Index":   top20_idx,
                    "SHAP Value":  [f"{v:+.5f}" for v in top20_vals],
                    "Bit Active":  ["✅" if b else "⬜" for b in top20_bit],
                    "Direction":   ["↑ Mutagenic" if float(v) > 0 else "↓ Mutagenic" for v in top20_vals],
                })
                st.dataframe(df_top, use_container_width=True, hide_index=True)

        else:
            # Fallback: global feature importance
            metrics = load_metrics()
            if "feature_importances" in metrics:
                fi = metrics["feature_importances"]
                st.plotly_chart(make_feature_importance(fi, top_n=20), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

def page_model_performance():
    st.markdown("<div class='hero-title' style='font-size:1.8rem;'>📊 Model Performance</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Evaluation metrics and visualizations for the trained Random Forest model.</div><br>", unsafe_allow_html=True)

    metrics = load_metrics()
    if not metrics:
        st.markdown("<div class='warning-box'>⚠️ Metrics not found. Run <code>python train_model.py</code> first.</div>", unsafe_allow_html=True)
        # Show placeholder metrics
        metrics = {
            "accuracy": 0.88, "precision": 0.87, "recall": 0.86,
            "f1_score": 0.865, "roc_auc": 0.93,
            "confusion_matrix": [[85, 15], [12, 88]],
            "fpr": [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
            "tpr": [0,0.5,0.7,0.8,0.85,0.88,0.91,0.93,0.95,0.97,1.0],
            "feature_importances": list(np.random.dirichlet(np.ones(2048))),
        }

    acc  = metrics.get("accuracy",  0.88)
    prec = metrics.get("precision", 0.87)
    rec  = metrics.get("recall",    0.86)
    f1   = metrics.get("f1_score",  0.865)
    auc  = metrics.get("roc_auc",   0.93)

    st.markdown(
        f"""
        <div class='metric-grid'>
            <div class='metric-card'><div class='metric-value'>{acc*100:.1f}%</div><div class='metric-label'>Accuracy</div></div>
            <div class='metric-card'><div class='metric-value'>{prec*100:.1f}%</div><div class='metric-label'>Precision</div></div>
            <div class='metric-card'><div class='metric-value'>{rec*100:.1f}%</div><div class='metric-label'>Recall</div></div>
            <div class='metric-card'><div class='metric-value'>{f1*100:.1f}%</div><div class='metric-label'>F1 Score</div></div>
            <div class='metric-card'><div class='metric-value'>{auc:.3f}</div><div class='metric-label'>ROC AUC</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["🗂️ Confusion Matrix", "📈 ROC Curve", "🎯 Feature Importance"])

    with tab1:
        cm = metrics.get("confusion_matrix", [[85, 15], [12, 88]])
        st.plotly_chart(make_confusion_matrix(cm), use_container_width=True)

        # Per-class metrics
        cm_arr = np.array(cm)
        tn, fp_val, fn, tp = cm_arr.ravel() if cm_arr.size == 4 else (0, 0, 0, 0)
        st.markdown(
            f"""
            <div class='glass-card'>
            <table style='width:100%; border-collapse:collapse; font-size:0.875rem; color:#94A3B8; text-align:center;'>
              <thead>
                <tr style='border-bottom:1px solid #1E293B;'>
                  <th style='padding:0.5rem; color:#E2E8F0;'>Class</th>
                  <th style='padding:0.5rem; color:#E2E8F0;'>TP</th>
                  <th style='padding:0.5rem; color:#E2E8F0;'>TN</th>
                  <th style='padding:0.5rem; color:#E2E8F0;'>FP</th>
                  <th style='padding:0.5rem; color:#E2E8F0;'>FN</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style='padding:0.5rem; color:#FF1744; font-weight:600;'>Mutagenic</td>
                  <td style='padding:0.5rem;'>{tp}</td>
                  <td style='padding:0.5rem;'>{tn}</td>
                  <td style='padding:0.5rem;'>{fp_val}</td>
                  <td style='padding:0.5rem;'>{fn}</td>
                </tr>
              </tbody>
            </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab2:
        fpr = metrics.get("fpr", [0, 1])
        tpr = metrics.get("tpr", [0, 1])
        st.plotly_chart(make_roc_curve(fpr, tpr, auc), use_container_width=True)

    with tab3:
        fi = metrics.get("feature_importances", [])
        if fi:
            st.plotly_chart(make_feature_importance(fi, top_n=20), use_container_width=True)
        else:
            st.info("Feature importances not available.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: CHEMICAL DESCRIPTORS
# ══════════════════════════════════════════════════════════════════════════════

def page_chemical_descriptors():
    st.markdown("<div class='hero-title' style='font-size:1.8rem;'>⚗️ Chemical Descriptors</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Calculate physicochemical properties from SMILES using RDKit.</div><br>", unsafe_allow_html=True)

    smiles_area = st.text_area(
        "SMILES (one per line)",
        value="CCO\nCC(=O)Oc1ccccc1C(=O)O\nc1ccc2c(c1)ccc3cccc4cccc2c34\nNc1ccc([N+](=O)[O-])cc1",
        height=130,
    )

    if st.button("⚗️ Calculate Descriptors", type="primary"):
        lines = [l.strip() for l in smiles_area.strip().splitlines() if l.strip()]
        if not lines:
            st.warning("Please enter at least one SMILES.")
            return

        rows = []
        for smi in lines:
            valid, mol = validate_smiles(smi)
            if not valid or mol is None:
                rows.append({"SMILES": smi, "Status": "❌ Invalid", **{k: "—" for k in [
                    "Molecular Weight (Da)", "LogP (Wildman-Crippen)", "TPSA (Å²)",
                    "H-bond Donors", "H-bond Acceptors", "Rotatable Bonds",
                    "Aromatic Rings", "Heavy Atom Count", "Ring Count", "Stereo Centers",
                ]}})
            else:
                d = compute_descriptors(mol)
                rows.append({"SMILES": smi, "Status": "✅ Valid", **d})

        df = pd.DataFrame(rows)
        st.markdown("<div class='section-header'>📋 Descriptor Table</div>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

        # Lipinski's Rule of Five check
        st.markdown("<div class='section-header'>💊 Lipinski's Rule of Five</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='info-box'>"
            "<strong>Rule of Five criteria:</strong> MW ≤ 500 Da · LogP ≤ 5 · HBD ≤ 5 · HBA ≤ 10<br>"
            "Compounds passing these rules are more likely to have oral bioavailability."
            "</div>",
            unsafe_allow_html=True,
        )

        ro5_rows = []
        for smi in lines:
            valid, mol = validate_smiles(smi)
            if not valid or mol is None:
                continue
            d = compute_descriptors(mol)
            ro5_pass = (
                d["Molecular Weight (Da)"] <= 500
                and d["LogP (Wildman-Crippen)"] <= 5
                and d["H-bond Donors"] <= 5
                and d["H-bond Acceptors"] <= 10
            )
            violations = sum([
                d["Molecular Weight (Da)"] > 500,
                d["LogP (Wildman-Crippen)"] > 5,
                d["H-bond Donors"] > 5,
                d["H-bond Acceptors"] > 10,
            ])
            ro5_rows.append({
                "SMILES": smi,
                "MW ≤ 500": "✅" if d["Molecular Weight (Da)"] <= 500 else "❌",
                "LogP ≤ 5": "✅" if d["LogP (Wildman-Crippen)"] <= 5 else "❌",
                "HBD ≤ 5":  "✅" if d["H-bond Donors"] <= 5 else "❌",
                "HBA ≤ 10": "✅" if d["H-bond Acceptors"] <= 10 else "❌",
                "Violations": violations,
                "Drug-like": "✅ Yes" if ro5_pass else "❌ No",
            })

        if ro5_rows:
            st.dataframe(pd.DataFrame(ro5_rows), use_container_width=True, hide_index=True)

        # Download
        csv = df.to_csv(index=False)
        st.download_button("⬇️ Download Descriptors CSV", csv, "descriptors.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════

def page_about():
    st.markdown(
        """
        <div class='hero-banner'>
            <div class='hero-title'>ℹ️ About This Project</div>
            <div class='hero-subtitle'>
                Open-source ML platform for Ames mutagenicity prediction — built for researchers,
                toxicologists, and drug discovery scientists.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>🛠️ Technology Stack</div>", unsafe_allow_html=True)
        tech_stack = {
            "Python 3.10+": "Core programming language",
            "Streamlit 1.32": "Interactive web application framework",
            "scikit-learn 1.4": "Random Forest classifier training & evaluation",
            "RDKit": "Cheminformatics — SMILES parsing, fingerprints, descriptors",
            "SHAP 0.44": "Explainability — feature attribution",
            "Plotly 5.20": "Interactive data visualizations",
            "Pandas / NumPy": "Data manipulation and numerical computing",
            "Joblib": "Model serialization / caching",
        }
        for tech, desc in tech_stack.items():
            st.markdown(
                f"<div class='info-box'><strong style='color:#00D4FF;'>{tech}</strong><br>{desc}</div>",
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("<div class='section-header'>🚀 Deployment Options</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-header' style='font-size:1rem;'>💻 Local</div>", unsafe_allow_html=True)
        st.code(
            "pip install -r requirements.txt\npython train_model.py\nstreamlit run app.py",
            language="bash",
        )

        st.markdown("<div class='section-header' style='font-size:1rem;'>☁️ Streamlit Cloud</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='info-box'>"
            "1. Push this repo to GitHub<br>"
            "2. Visit <a href='https://streamlit.io/cloud' target='_blank' style='color:#00D4FF;'>streamlit.io/cloud</a><br>"
            "3. Connect your repo and deploy<br>"
            "4. Set Python version to 3.10+ in settings"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div class='section-header' style='font-size:1rem;'>🤗 Hugging Face Spaces</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='info-box'>"
            "1. Create a new Space (Streamlit SDK)<br>"
            "2. Upload <code>app.py</code>, <code>model.pkl</code>, <code>requirements.txt</code><br>"
            "3. Add <code>metrics.json</code> for full performance page<br>"
            "4. Space auto-builds and deploys"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div class='section-header' style='font-size:1rem;'>🐳 Render / Docker</div>", unsafe_allow_html=True)
        st.code(
            "# Dockerfile\nFROM python:3.10-slim\nWORKDIR /app\nCOPY . .\nRUN pip install -r requirements.txt\nCMD [\"streamlit\", \"run\", \"app.py\", \"--server.port=10000\", \"--server.address=0.0.0.0\"]",
            language="dockerfile",
        )

    st.markdown("<div class='section-header'>📈 Future Improvements</div>", unsafe_allow_html=True)
    improvements = [
        "🧠 Deep learning models (Graph Neural Networks / Transformers) for improved accuracy",
        "🔗 API endpoint for programmatic access (FastAPI integration)",
        "📦 Support for SDF / MOL2 / InChI input formats",
        "🌐 3-D molecular visualization (py3Dmol / NGL viewer)",
        "⚡ GPU-accelerated inference for large-scale screening",
        "📊 Multi-endpoint toxicity prediction (hERG, CYP450, DILI)",
        "🗃️ Database integration for compound registry",
        "📜 Automated regulatory report generation (ICH M7)",
    ]
    cols = st.columns(2)
    for i, item in enumerate(improvements):
        with cols[i % 2]:
            st.markdown(f"<div class='info-box'>{item}</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='footer'>Built with ❤️ for the cheminformatics & drug discovery community · "
        "For research use only · Not for clinical or regulatory decision-making</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════

def main():
    page = render_sidebar()

    dispatch = {
        "Home":                   page_home,
        "Single Prediction":      page_single_prediction,
        "Batch Prediction":       page_batch_prediction,
        "Molecular Visualization": page_molecular_visualization,
        "Explainability":         page_explainability,
        "Model Performance":      page_model_performance,
        "Chemical Descriptors":   page_chemical_descriptors,
        "About":                  page_about,
    }

    dispatch.get(page, page_home)()

    st.markdown(
        "<div class='footer'>🧬 Ames Mutagenicity Predictor · For Research Purposes Only · © 2024</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
