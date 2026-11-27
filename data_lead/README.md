## Data Lead (Ayesha): cleaning, EDA, leakage scan

Notebook: `notebooks/02_data_lead.ipynb` (runs top to bottom in Colab, seed 42). Outputs are in `data/` and `results/`.

**Method:**
- Loaded the PhiUSIIL dataset (UCI id 967): 235,795 rows x 55 columns, no missing values.
- Duplicates: 0 duplicate rows but 425 duplicate URLs (all phishing, none with conflicting labels). Removed them, keeping the first: 235,370 rows (134,850 legitimate, 100,520 phishing, i.e. 57.3% / 42.7%).
- Columns: 4 text (`URL`, `Domain`, `TLD`, `Title`) and 50 numeric features. 19 of them can be computed from the URL alone (`results/feature_table.csv`).
- EDA: class means per feature, histograms, correlation heatmap (only 2 pairs above 0.9), Mann-Whitney test on URL length.
- Leakage scan: depth-3 decision tree on each single feature, 5-fold CV AUC; suspects are features above 0.95 (`results/leakage_scan.csv`).
- Baselines: decision tree (max_depth = 5), 80/20 stratified split.

**Main results:**
- Phishing URLs are longer (median 34 vs 26 characters, p < 0.001) and contain more digits (4.3 vs 0.05 on average).
- 7 leakage suspects found: `URLSimilarityIndex`, `LineOfCode`, `NoOfExternalRef`, `NoOfImage`, `NoOfJS`, `NoOfSelfRef`, `NoOfCSS`.
- Decision tree on all 50 features scores 1.0 (leakage). With the 7 suspects removed (43 features): accuracy 0.9860, ROC-AUC 0.9952. URL-only (19 features): accuracy 0.9806, ROC-AUC 0.9903.
- Removing `IsHTTPS` from the URL-only set drops CV AUC from 0.9907 to 0.8556, so HTTPS is a very strong signal in this data.
- All legitimate URLs use HTTPS, compared with 49% of phishing URLs.

**Limitations:** the leakage threshold (0.95) is my choice; single-feature scans can miss features that leak only in combination; `data_dictionary.csv` still has an empty "meaning" column.
