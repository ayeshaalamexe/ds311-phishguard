## Product Lead (Hassaan): URL-text model, robustness test and app

Notebook: `notebooks/04_product_lead.ipynb` (runs top to bottom in Colab, seed 42). Files are in `product/`.

- URL-text model on 33 features built from the raw URL string; model used in the app: **both**.
- 5-fold CV: ROC-AUC 0.9990, phishing recall 0.9956, about 4.4 missed phishing and 0.0 false alarms per 1,000.
- Robustness: worst change is "force https" (65.4% of phishing still caught).
- Run the app: `pip install -r product/requirements.txt` then `streamlit run product/app.py`.
- URLs are only treated as text and are never opened.
