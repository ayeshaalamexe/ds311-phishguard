# PhishGuard: Streamlit app
# The app only reads the URL as TEXT. It never opens or visits a URL.
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent      # folder that holds app.py (repo root on Streamlit Cloud)

# Files may sit next to app.py, inside product/, or one level up.
# locate() checks each place so the app works whichever layout the repo has.
SEARCH_DIRS = []
for d in [BASE, BASE / "product", BASE.parent]:
    if d not in SEARCH_DIRS:
        SEARCH_DIRS.append(d)
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))


def locate(*parts):
    for d in SEARCH_DIRS:
        p = d.joinpath(*parts)
        if p.exists():
            return p
    return BASE.joinpath(*parts)            # not found: return the default path


from url_features import (FEATURE_NAMES, FEATURE_TEXT, normalize_url, url_features,
                          predict_scores, explain_features, explain_text, template_notes)

st.set_page_config(page_title="PhishGuard", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load(locate("models", "phishguard_bundle.joblib"))


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


def show_csv_if_exists(rel, title):
    path = locate("results", rel)
    if path.exists():
        st.subheader(title)
        st.dataframe(load_csv(path), hide_index=True)


def show_image_if_exists(rel, caption):
    path = locate("results", rel)
    if path.exists():
        st.image(str(path), caption=caption)


try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error("Model file not found: models/phishguard_bundle.joblib. "
             "Check that the models folder is in the repo.")
    st.stop()

st.title("🛡️ PhishGuard: Phishing URL Miner")
st.caption("Paste a URL to get a risk score. PhishGuard only reads the text of the URL. "
           "It never opens the link.")

tab1, tab2, tab3, tab4 = st.tabs(["Check a URL", "Dashboard", "Cluster explorer", "About"])

# ------------------------------------------------------------------ tab 1
with tab1:
    raw = st.text_input("URL to check", placeholder="https://example.com/login")
    go = st.button("Check URL")

    if go:
        if raw.strip() == "":
            st.warning("Please paste a URL first.")
        elif len(raw) > 2000:
            st.warning("That URL is too long (more than 2000 characters).")
        else:
            url, assumed = normalize_url(raw)
            if assumed:
                st.info("No http:// or https:// was typed, so https:// was assumed.")
            st.write("Checked text (not opened):")
            st.code(url, language=None)

            score = float(predict_scores([url], bundle)[0])
            if score >= 0.7:
                st.error(f"HIGH risk: score {score * 100:.1f} / 100. Do not enter passwords here.")
            elif score >= 0.3:
                st.warning(f"MEDIUM risk: score {score * 100:.1f} / 100. Be careful.")
            else:
                st.success(f"LOW risk: score {score * 100:.1f} / 100.")
            st.progress(min(max(score, 0.0), 1.0))
            st.caption("The score is a model estimate, not a guarantee. A low score does not "
                       "prove that a site is safe.")

            notes = template_notes(url)
            if notes:
                with st.expander("How much should you trust this score? (read this)",
                                 expanded=score >= 0.3):
                    st.write("The training data (PhiUSIIL) has a very narrow idea of a legitimate URL: "
                             "**https://www.name.tld** and nothing else. This URL differs from that shape:")
                    for note in notes:
                        st.write("- " + note)
                    st.write("So a high score here can mean *'this does not look like a bare home page'* "
                             "and not *'this is phishing'*. See the Dashboard tab, 'Shortcut check'.")

            if bundle["use"] in ("features", "both"):
                st.subheader("URL traits behind the score")
                table = explain_features(url, bundle)
                table["effect_on_score"] = table["effect_on_score"].round(2)
                up = table[table["effect_on_score"] > 0.05].head(5)
                down = table[table["effect_on_score"] < -0.05].sort_values("effect_on_score").head(3)
                rename = {"meaning": "trait", "value": "this URL",
                          "typical_legit_value": "typical legitimate URL",
                          "effect_on_score": "push (log-odds)"}
                cols = ["meaning", "value", "typical_legit_value", "effect_on_score"]
                if len(up) > 0:
                    st.write("**Raise the risk:**")
                    st.dataframe(up[cols].rename(columns=rename), hide_index=True)
                else:
                    st.write("No single trait raises the risk by much.")
                if len(down) > 0:
                    st.write("**Lower the risk:**")
                    st.dataframe(down[cols].rename(columns=rename), hide_index=True)
                st.caption("How this is computed: starting from a typical legitimate URL, the traits are "
                           "switched to this URL's values in random order (40 times) and we average how "
                           "much each switch moves the feature model's log-odds (sampled Shapley values). "
                           "Positive = towards phishing. The pushes add up to the whole score.")

            if bundle["use"] in ("tfidf", "both"):
                st.subheader("Text pieces that look like phishing")
                clues = explain_text(url, bundle)
                if clues:
                    for piece, value in clues:
                        st.write(f"`{piece}`  (push: {value:.2f})")
                else:
                    st.write("No text piece pushes the score towards phishing.")

            with st.expander("All computed features of this URL"):
                feats = url_features(url)
                st.dataframe(pd.DataFrame({"feature": [FEATURE_TEXT[k] for k in FEATURE_NAMES],
                                           "value": [round(feats[k], 3) for k in FEATURE_NAMES]}),
                             hide_index=True)

# ------------------------------------------------------------------ tab 2
with tab2:
    st.header("Feature distributions")
    sample_path = locate("data", "sample_urls.csv")
    if sample_path.exists():
        sample = load_csv(sample_path)
        sample["class"] = sample["is_phish"].map({0: "legitimate", 1: "phishing"})
        st.write(f"A random sample of {len(sample):,} URLs from the PhiUSIIL dataset.")

        choice = st.selectbox("Pick a feature", FEATURE_NAMES,
                              format_func=lambda k: FEATURE_TEXT[k])
        upper = sample[choice].quantile(0.99)          # cut the extreme 1% so the plot is readable
        fig, ax = plt.subplots(figsize=(7, 3.5))
        for name, color in [("legitimate", "tab:blue"), ("phishing", "tab:red")]:
            values = sample.loc[sample["class"] == name, choice].clip(upper=upper)
            ax.hist(values, bins=30, alpha=0.55, density=True, label=name, color=color)
        ax.set_xlabel(FEATURE_TEXT[choice] + " (top 1% cut off)")
        ax.set_ylabel("density")
        ax.legend()
        st.pyplot(fig)

        st.write("Average of each feature by class")
        means = sample.groupby("class")[FEATURE_NAMES].mean().T.round(3)
        means.index = [FEATURE_TEXT[k] for k in means.index]
        st.dataframe(means)
    else:
        st.info("data/sample_urls.csv not found.")

    st.header("Model results")
    show_csv_if_exists("url_model_comparison.csv", "URL-text models (5-fold cross-validation)")
    show_csv_if_exists("threshold_table.csv", "Choosing the alert threshold")
    show_csv_if_exists("robustness_table.csv", "Robustness: phishing URLs changed to evade the model")
    show_csv_if_exists("feature_ranking_url_text.csv", "Which URL-text features carry the most information")
    show_csv_if_exists("template_share.csv", "What a 'legitimate' URL looks like in the training data")
    show_csv_if_exists("shortcut_check.csv", "Shortcut check: how much of the score is just the URL's shape?")
    show_csv_if_exists("realworld_sanity.csv",
                       "Sanity check on well-known real websites (small, hand-picked, illustrative)")
    col1, col2 = st.columns(2)
    with col1:
        show_image_if_exists("confusion_roc_final.png", "Final model, out-of-fold predictions")
    with col2:
        show_image_if_exists("topk_curve.png", "How many features are needed (Mining Lead)")

# ------------------------------------------------------------------ tab 3
with tab3:
    st.header("Cluster explorer")
    st.write("K-means groups of phishing URLs found by the Mining Lead (3 loose groups, "
             "silhouette 0.245, so the groups overlap).")
    cluster_path = locate("results", "cluster_profiles.csv")
    if cluster_path.exists():
        clusters = load_csv(cluster_path)
        pick = st.selectbox("Cluster", clusters["cluster_name"].tolist())
        row = clusters[clusters["cluster_name"] == pick].iloc[0]
        share = 100 * row["size"] / clusters["size"].sum()
        st.write(f"**{int(row['size']):,} phishing URLs** ({share:.1f}% of phishing URLs)")

        show_cols = ["URLLength", "DomainLength", "NoOfSubDomain", "NoOfDegitsInURL",
                     "SpacialCharRatioInURL", "IsHTTPS"]
        compare = clusters.set_index("cluster_name")[show_cols].T
        st.write("Average values (all clusters side by side)")
        st.dataframe(compare)
        st.write("Most common endings (TLDs): " + str(row["top_TLDs"]))
        st.write("Example URLs (shown as text only):")
        for u in str(row["example_urls"]).split(" | "):
            st.code(u, language=None)
        show_image_if_exists("clusters_pca.png", "Clusters in a 2D PCA view")
    else:
        st.info("results/cluster_profiles.csv not found.")

# ------------------------------------------------------------------ tab 4
with tab4:
    st.header("About PhishGuard")
    st.write(
        "PhishGuard is a DS-311 Data Mining project. It reads only the text of a URL "
        "(length, digits, special characters, sub-domains, character patterns) and gives a risk score."
    )
    st.write(f"Scoring model in use: **{bundle['use']}**")
    st.subheader("Limits you should know")
    st.write(
        "- The training data (PhiUSIIL) has mostly short home-page URLs as legitimate examples, "
        "so long but harmless links (for example a deep link on a big site) can get a higher score.\n"
        "- In this dataset every legitimate URL has the exact shape https://www.name.tld with no path, "
        "no query and no trailing slash, so any other shape (http://, no www., a path) raises the score a lot. "
        "Much of the very high cross-validation score comes from this shape and not from real phishing clues.\n"
        "- Phishers can change their URLs to avoid detection. See the robustness table in the Dashboard tab.\n"
        "- The score is only an estimate. Do not use it as the only protection."
    )
    st.write("Safety rule: URLs are treated as text only and are never opened.")
