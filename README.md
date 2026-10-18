# ds311-phishguard
DS-311 Data Mining project: PhishGuard, Phishing URL Miner

## Team and Contribution Log
- Ayesha: Data lead
- Sara: Mining lead
- Hassaan: Product lead
## Mining Lead (Sara): feature selection, model comparison, leakage check, clustering

Notebook: `notebooks/03_mining_lead.ipynb` (runs top to bottom in Colab, seed 42). Outputs are in `results/`.

**Method (43 features left after removing 7 leakage suspects; 19 are URL-only):**
- Feature selection: mutual information, chi-square and random forest importance, combined by average rank (`results/feature_ranking.csv`, `results/topk_curve.csv`).
- Models: decision tree and Naive Bayes baselines vs random forest, gradient boosting and SVM, stratified 5-fold CV with scaling inside a Pipeline (`results/model_comparison.csv`). SVM used a 10,000-row stratified sample because it is slow; the other models used 50,000 rows.
- Leakage check: with/without suspect features, with/without IsHTTPS, and domain-grouped CV (`results/leakage_check.csv`).
- Clustering: K-means on the 19 URL features of phishing URLs, standardised and clipped to +/-3, k chosen by silhouette (`results/cluster_profiles.csv`).

**Main results (5-fold CV):**
- Top 8 URL-only features, random forest: ROC-AUC 0.9973, phishing recall 0.9927.
- URL-only gradient boosting (19 features): AUC 0.998, about 5.6 missed phishing URLs and 0.9 false alarms per 1,000.
- Decision tree (URL-only) misses about 41 per 1,000 and Naive Bayes about 234 per 1,000.
- Domain-grouped CV gave almost the same scores as normal CV. Few domains repeat in the data, so this test is limited.
- Page-content features still reach AUC 1.000 after removing the 7 suspects, so the URL-only score is the fair headline.
- Phishing URLs form 3 loose groups (silhouette 0.245): short plain HTTP URLs, medium HTTPS URLs, very long parameter-heavy URLs.

**Limitations:** the dataset is easy (legitimate URLs are mostly short homepages), cluster structure is weak, and clusters were not checked against real campaigns.
