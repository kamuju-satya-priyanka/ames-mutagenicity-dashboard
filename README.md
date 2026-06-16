# 🧬 Ames Mutagenicity Predictor using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)](https://scikit-learn.org)
[![RDKit](https://img.shields.io/badge/RDKit-2023.9-green)](https://rdkit.org)
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

> **An AI-powered platform for rapid, in silico Ames mutagenicity screening of chemical compounds.**

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Information](#-model-information)
- [Deployment](#-deployment)
- [Screenshots](#-screenshots)
- [Future Improvements](#-future-improvements)
- [Disclaimer](#-disclaimer)

---

## 🧬 Project Overview

The **Ames Mutagenicity Predictor** is a full-stack machine learning web application that predicts whether a chemical compound is **Mutagenic** or **Non-Mutagenic** based on its SMILES (Simplified Molecular Input Line Entry System) representation.

The Ames test (Salmonella mutagenicity assay) is a widely used biological assay to assess the mutagenic potential of chemical substances. This application provides an *in silico* alternative, significantly reducing time and cost compared to wet-lab testing.

### Key Highlights
- 🎯 **88%+ Accuracy** on test set
- ⚡ **Real-time predictions** from SMILES input
- 📊 **SHAP-based explanations** for every prediction
- 📂 **Batch screening** via CSV upload
- 🔭 **2D molecular structure** visualization
- 💊 **Physicochemical descriptor** calculation
- 🏥 **Toxicity risk gauge** with safe/moderate/high tiers

---

## ✨ Features

| Page | Description |
|------|-------------|
| 🏠 **Home** | Project overview, dataset info, model summary, accuracy metrics |
| 🔬 **Single Prediction** | Predict mutagenicity of a single SMILES; gauge chart + probability bar |
| 📂 **Batch Prediction** | Upload CSV of SMILES, download results with predictions & probabilities |
| 🧪 **Molecular Visualization** | 2D structure rendering using RDKit SVG |
| 🤖 **Explainability (XAI)** | SHAP feature impact, waterfall plot, top contributing fingerprint bits |
| 📊 **Model Performance** | Confusion matrix, ROC curve, feature importance chart |
| ⚗️ **Chemical Descriptors** | MW, LogP, TPSA, HBD, HBA, rotatable bonds, Lipinski's Rule of Five |
| ℹ️ **About** | Tech stack, deployment options, future roadmap |

---

## 🛠️ Technology Stack

| Component | Library | Version |
|-----------|---------|---------|
| Web Framework | Streamlit | 1.32.0 |
| Machine Learning | scikit-learn | 1.4.1 |
| Cheminformatics | RDKit | 2023.9.5 |
| Explainability | SHAP | 0.44.1 |
| Visualization | Plotly | 5.20.0 |
| Data Processing | Pandas / NumPy | 2.2.1 / 1.26.4 |
| Model Serialization | Joblib | 1.3.2 |
| Image Processing | Pillow | 10.2.0 |

---

## 📁 Project Structure

```
ames-mutagenicity-predictor/
├── app.py              # Main Streamlit application (all pages)
├── train_model.py      # Model training script
├── model.pkl           # Trained Random Forest model (generated)
├── metrics.json        # Model performance metrics (generated)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/ames-mutagenicity-predictor.git
cd ames-mutagenicity-predictor
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note for Windows users:** RDKit may require extra steps. Install via conda if pip fails:
> ```bash
> conda install -c conda-forge rdkit
> ```

### Step 4: Train the Model
```bash
python train_model.py
```
This generates `model.pkl` and `metrics.json` in the project directory.

### Step 5: Launch the Application
```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## 💡 Usage

### Single Compound Prediction

1. Navigate to **Single Prediction** in the sidebar.
2. Click an example compound or type your own SMILES string.
3. Click **⚡ Predict Mutagenicity**.
4. Review:
   - Mutagenic / Non-Mutagenic badge
   - Probability bar (mutagenic vs safe)
   - Toxicity Risk Gauge (0–100%)
   - 2D molecular structure
   - Chemical descriptors

**Example SMILES:**

| Compound | SMILES | Expected |
|----------|--------|----------|
| Ethanol | `CCO` | Non-Mutagenic |
| Benzo[a]pyrene | `c1ccc2c(c1)ccc3cccc4cccc2c34` | Mutagenic |
| Aspirin | `CC(=O)Oc1ccccc1C(=O)O` | Non-Mutagenic |
| 4-Nitroaniline | `Nc1ccc([N+](=O)[O-])cc1` | Mutagenic |

### Batch Prediction

1. Navigate to **Batch Prediction**.
2. Download the CSV template.
3. Fill in your SMILES (one per row under the `SMILES` column).
4. Upload the CSV.
5. View results table and summary pie chart.
6. Download the results CSV.

**Input CSV format:**
```csv
SMILES
CCO
CCN
c1ccc2c(c1)ccc3cccc4cccc2c34
```

**Output CSV columns:**
```
SMILES, Valid, Prediction, Mutagenic_Probability, Non_Mutagenic_Probability, Confidence
```

### Explainability

1. Go to **Explainability (XAI)**.
2. Enter a SMILES string.
3. Click **🔍 Explain Prediction**.
4. Explore three tabs:
   - **Feature Impact** — SHAP bar chart
   - **SHAP Waterfall** — cumulative contribution plot
   - **Top Features** — ranked fingerprint bit table

---

## 🧠 Model Information

### Algorithm
**Random Forest Classifier** (`scikit-learn`)

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| `n_estimators` | 200 |
| `max_features` | sqrt |
| `class_weight` | balanced |
| `random_state` | 42 |
| `n_jobs` | -1 (all cores) |

### Molecular Representation
- **Type:** Morgan Fingerprints (ECFP4 equivalent)
- **Radius:** 2
- **Bits:** 2048

### Performance Metrics
| Metric | Score |
|--------|-------|
| Accuracy | ~88% |
| Precision | ~87% |
| Recall | ~86% |
| F1 Score | ~86.5% |
| ROC AUC | ~0.93 |

---

## 🌐 Deployment

### Local Development
```bash
streamlit run app.py
```

### Streamlit Community Cloud (Free)
1. Push code to a public GitHub repository.
2. Visit [streamlit.io/cloud](https://streamlit.io/cloud).
3. Click **New App** → connect your repo.
4. Set **Main file path** to `app.py`.
5. Click **Deploy**.

### Hugging Face Spaces
1. Create a new Space with **Streamlit** SDK.
2. Upload `app.py`, `train_model.py`, `model.pkl`, `metrics.json`, `requirements.txt`.
3. Space builds automatically.

### Render
1. Create a `render.yaml`:
```yaml
services:
  - type: web
    name: ames-predictor
    env: python
    buildCommand: pip install -r requirements.txt && python train_model.py
    startCommand: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```
2. Connect your GitHub repo on [render.com](https://render.com).

### Docker
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python train_model.py

EXPOSE 8501
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

```bash
docker build -t ames-predictor .
docker run -p 8501:8501 ames-predictor
```

---

## 📸 Screenshots

> *Screenshots placeholder — replace with actual screenshots after deployment.*

| Page | Description |
|------|-------------|
| ![Home Page](screenshots/home.png) | Home dashboard with model metrics |
| ![Single Prediction](screenshots/prediction.png) | SMILES input with result badge & gauge |
| ![Batch Prediction](screenshots/batch.png) | CSV upload and results table |
| ![Explainability](screenshots/shap.png) | SHAP waterfall and feature impact |
| ![Model Performance](screenshots/performance.png) | ROC curve and confusion matrix |

---

## 🔮 Future Improvements

- [ ] **Graph Neural Networks** — GCN / MPNN for higher accuracy
- [ ] **Multi-endpoint prediction** — hERG, CYP450, DILI, Skin Sensitization
- [ ] **3D molecular visualization** — py3Dmol integration
- [ ] **SDF / MOL2 / InChI support** — extended input formats
- [ ] **REST API** — FastAPI wrapper for programmatic access
- [ ] **Active learning** — model improvement with user feedback
- [ ] **Regulatory report** — Auto-generated ICH M7 compliant PDF reports
- [ ] **Database integration** — ChEMBL / PubChem compound registry lookup

---

## 📜 Disclaimer

> ⚠️ **This tool is for research and educational purposes only.**
>
> Predictions made by this application are based on a machine learning model trained on computational data. They should **not** be used as a sole basis for toxicological decision-making, regulatory submissions, or clinical applications. Always validate predictions with appropriate experimental assays and consult a qualified toxicologist.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ for the cheminformatics & drug discovery community<br>
  <strong>🧬 Ames Mutagenicity Predictor</strong>
</p>
