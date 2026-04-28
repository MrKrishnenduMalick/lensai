# ⚖️ FairLens AI – Transparent Intelligence

Detect bias in ML models · Explain predictions with SHAP · Build trustworthy AI

## 🚀 Quick Start (Local)

```bash
cd fairlens_ai
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy on Streamlit Cloud

1. Push `fairlens_ai/` folder to GitHub repo
2. https://share.streamlit.io → New app → set Main file: `app.py`
3. Deploy ✓

## 📂 Structure

```
fairlens_ai/
├── app.py
├── requirements.txt
├── sample_data.csv
├── .streamlit/config.toml   ← light theme
└── utils/
    ├── __init__.py           ← REQUIRED (makes utils a package)
    ├── model.py
    ├── bias.py
    └── explain.py
```

## ⚠️ Fix for ModuleNotFoundError: No module named 'utils'

Run from INSIDE fairlens_ai/ directory:
```bash
cd fairlens_ai
streamlit run app.py          # ✅ correct
```
NOT:
```bash
streamlit run fairlens_ai/app.py  # ❌ wrong — utils/ not in path
```
