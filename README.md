# ds311-phishguard

DS-311 Data Mining project: PhishGuard, Phishing URL Miner

## Team and Contribution Log

* Ayesha: Data lead
* Sara: Mining lead
* Hassaan: Product lead

## Mining Lead (Sara): feature selection, model comparison, leakage check, clustering

Notebook: `notebooks/03\_mining\_lead.ipynb` (runs top to bottom in Colab, seed 42). Outputs are in `results/`.

**Method (43 features left after removing 7 leakage suspects; 19 are URL-only):**

* Feature selection: mutual information, chi-square and random forest importance, combined by average rank (`results/feature\_ranking.csv`, `results/topk\_curve.csv`).
* Models: decision tree and Naive Bayes baselines vs random forest, gradient boosting and SVM, stratified 5-fold CV with scaling inside a Pipeline (`results/model\_comparison.csv`). SVM used a 10,000-row stratified sample because it is slow; the other models used 50,000 rows.
* Leakage check: with/without suspect features, with/without IsHTTPS, and domain-grouped CV (`results/leakage\_check.csv`).
* Clustering: K-means on the 19 URL features of phishing URLs, standardised and clipped to +/-3, k chosen by silhouette (`results/cluster\_profiles.csv`).

**Main results (5-fold CV):**

* Top 8 URL-only features, random forest: ROC-AUC 0.9973, phishing recall 0.9927.
* URL-only gradient boosting (19 features): AUC 0.998, about 5.6 missed phishing URLs and 0.9 false alarms per 1,000.
* Decision tree (URL-only) misses about 41 per 1,000 and Naive Bayes about 234 per 1,000.
* Domain-grouped CV gave almost the same scores as normal CV. Few domains repeat in the data, so this test is limited.
* Page-content features still reach AUC 1.000 after removing the 7 suspects, so the URL-only score is the fair headline.
* Phishing URLs form 3 loose groups (silhouette 0.245): short plain HTTP URLs, medium HTTPS URLs, very long parameter-heavy URLs.
* Tuning (URL-only): number of trees barely matters (AUC about 0.998); tree depth matters (RF depth 5 misses about 18 per 1,000, unlimited depth about 5.6). Gradient boosting is stable across learning rates and iterations.
* Error analysis: out-of-fold gradient boosting missed 120 of 21,354 phishing URLs; the missed ones look like short ordinary HTTPS homepages (`results/missed\_phishing\_examples.csv`).
* Naive Bayes recall is 0.44 unscaled and 0.82 scaled, so scaling explains the gap to the proposal's 0.41.
* Saved model: `models/url\_only\_gb.joblib` (gradient boosting on the 19 URL features, features in `models/url\_only\_features.json`), trained on all cleaned rows; its performance is the cross-validation score, not a score on its own training data.

**Limitations:** the dataset is easy (legitimate URLs are mostly short homepages), cluster structure is weak, and clusters were not checked against real campaigns.
# Product Lead (Hassaan): URL-text model, robustness test and app

Notebook: `notebooks/04\_product\_lead.ipynb` (runs top to bottom in Colab, seed 42). Files are in `product/`.
URL-text model on 33 features built from the raw URL string; model used in the app: \*\*both\*\*.
5-fold CV: ROC-AUC 0.9990, phishing recall 0.9956, about 4.4 missed phishing and 0.0 false alarms per 1,000.
Robustness: worst change is "force https" (65.4% of phishing still caught).
Run the app: `pip install -r product/requirements.txt` then `streamlit run product/app.py`.
URLs are only treated as text and are never opened.



