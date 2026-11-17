# PhishGuard: Product Lead report (Hassaan)

## 1. What I built
- A URL-text model that works on any pasted URL, using only the text of the URL.
- A Streamlit app: risk score, the URL traits behind the score, a feature dashboard, and a cluster explorer.
- A robustness test with changed (evasive) phishing URLs.

## 2. Method
- Data: PhiUSIIL, cleaned as in the Data Lead notebook (235,370 rows, 100,520 phishing and 134,850 legitimate).
- Feature model: 33 features computed from the URL string (length, digits, special characters, sub-domains, entropy, suspicious words, https, and others) with gradient boosting.
- Text model: TF-IDF on character 2- to 4-grams (50,000 n-grams) with logistic regression.
- Evaluation: stratified 5-fold cross-validation on a 50,000-row stratified sample (seed 42); the TF-IDF step is fitted inside each training fold.
- The final models are trained on all cleaned rows. Their performance is the cross-validation score, not a score on training data.

## 3. Results (5-fold cross-validation, threshold 0.5)
| model | roc_auc | precision | recall | missed_phish_per_1000 | false_alarms_per_1000 |
|---|---|---|---|---|---|
| Feature model (GB) | 0.9982 | 0.9998 | 0.9955 | 4.4956 | 0.1745 |
| Feature model, no is_https | 0.9986 | 0.9992 | 0.9949 | 5.0576 | 0.6284 |
| Text model (TF-IDF + LR) | 0.9989 | 1.0 | 0.9916 | 8.4293 | 0.0349 |
| Text model, no scheme | 0.9357 | 0.9396 | 0.8153 | 184.6961 | 39.063 |
| Both (average) | 0.999 | 1.0 | 0.9956 | 4.4488 | 0.0 |

Missed phishing and false alarms are counted per 1,000 phishing and per 1,000 legitimate URLs.
Model used in the app: **both** (ROC-AUC 0.9990, phishing recall 0.9956).

Shortcut check: without the scheme, the feature model has ROC-AUC 0.9986 (recall 0.9949) and the text model has ROC-AUC 0.9357 (recall 0.8153).

## 4. Threshold
| threshold | phishing_caught_percent | missed_phish_per_1000 | false_alarms_per_1000 | precision_percent |
|---|---|---|---|---|
| 0.1 | 99.63 | 3.7 | 1.3 | 99.83 |
| 0.2 | 99.58 | 4.2 | 0.3 | 99.95 |
| 0.3 | 99.57 | 4.3 | 0.1 | 99.99 |
| 0.5 | 99.56 | 4.4 | 0.0 | 100.0 |
| 0.7 | 99.19 | 8.1 | 0.0 | 100.0 |
| 0.9 | 98.68 | 13.2 | 0.0 | 100.0 |

A lower threshold catches more phishing but raises more false alarms. The app uses 0.3 and 0.7 as the medium and high risk levels.

## 5. Robustness
On the original test URLs the final model catches 99.7% of phishing URLs and flags 0.0% of legitimate URLs.
The change that hurts most is "force https": the model then catches only 65.4% of phishing URLs.

| change | phishing_still_caught_percent | legit_flagged_percent |
|---|---|---|
| none (original URLs) | 99.7 | 0.0 |
| force https | 65.4 | 0.0 |
| homepage only (drop path and query) | 98.4 | 0.0 |
| add www. | 98.8 | 0.0 |
| extra sub-domains (secure.login.) | 100.0 | 100.0 |
| homoglyphs (o to 0, l to 1 in domain) | 99.7 | 0.0 |
| add random query (?id=...) | 99.7 | 0.1 |
| toggle trailing slash | 99.8 | 99.5 |
| link shortener (bit.ly/...) | 100.0 | 100.0 |

Read both columns together: a change that raises the catch rate but also flags many legitimate URLs only shows that the model flags anything unusual. It is not real robustness.

## 6. Limits
- Legitimate URLs in this dataset are mostly short home pages, so the task is easier than real life, and long harmless links can get high scores.
- Shortcuts such as https can inflate scores, so we report results with and without the scheme.
- The clusters from the Mining Lead are weak (silhouette 0.245) and were not checked against real campaigns.
- The score is an estimate and should not be the only protection.
