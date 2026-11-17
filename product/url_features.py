"""URL feature code shared by the notebook and the Streamlit app.
Everything here works on the URL TEXT only. A URL is never opened or visited."""
import math
import re

import numpy as np
import pandas as pd

COMMON_TLDS = {"com", "org", "net", "edu", "gov"}
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
              "buff.ly", "cutt.ly", "rebrand.ly", "shorturl.at"}
SUSPICIOUS_WORDS = ["login", "signin", "verify", "secure", "account", "update",
                    "bank", "confirm", "password", "paypal", "wallet", "webscr",
                    "free", "bonus"]

# the order of this list is the column order used by the model
FEATURE_NAMES = [
    "url_length", "domain_length", "is_ip", "tld_length", "num_subdomains",
    "num_dots_in_host", "num_hyphens_in_host", "num_digits_in_host",
    "longest_host_label", "num_letters", "letter_ratio", "num_digits",
    "digit_ratio", "num_equals", "num_qmark", "num_ampersand", "num_at",
    "num_other_special", "special_ratio", "num_slashes", "path_length",
    "query_length", "has_query", "num_percent", "has_obfuscation",
    "is_https", "has_www", "has_port", "entropy", "longest_digit_run",
    "num_suspicious_words", "is_shortener", "common_tld",
]

# plain-English names, used in the app
FEATURE_TEXT = {
    "url_length": "URL length",
    "domain_length": "Domain length",
    "is_ip": "Domain is an IP address",
    "tld_length": "Length of the ending (TLD)",
    "num_subdomains": "Number of sub-domains",
    "num_dots_in_host": "Dots in the domain",
    "num_hyphens_in_host": "Hyphens in the domain",
    "num_digits_in_host": "Digits in the domain",
    "longest_host_label": "Longest part of the domain",
    "num_letters": "Number of letters",
    "letter_ratio": "Share of letters",
    "num_digits": "Number of digits",
    "digit_ratio": "Share of digits",
    "num_equals": "Number of '=' signs",
    "num_qmark": "Number of '?' signs",
    "num_ampersand": "Number of '&' signs",
    "num_at": "Number of '@' signs",
    "num_other_special": "Other special characters",
    "special_ratio": "Share of non-letter, non-digit characters",
    "num_slashes": "Number of '/' signs",
    "path_length": "Path length",
    "query_length": "Query (after '?') length",
    "has_query": "Has a query part",
    "num_percent": "Number of '%' (encoded characters)",
    "has_obfuscation": "Has encoded characters",
    "is_https": "Uses https",
    "has_www": "Starts with www.",
    "has_port": "Has a port number",
    "entropy": "Randomness of characters (entropy)",
    "longest_digit_run": "Longest run of digits",
    "num_suspicious_words": "Suspicious words (login, verify, ...)",
    "is_shortener": "Is a link shortener",
    "common_tld": "Ends in .com/.org/.net/.edu/.gov",
}


def normalize_url(url):
    """Strip spaces. If no http(s):// was typed, assume https://."""
    url = str(url).strip()
    assumed = False
    if "://" not in url:
        url = "https://" + url
        assumed = True
    return url, assumed


def split_url(url):
    """Split a URL string into scheme, host, port and the rest (path + query)."""
    if "://" in url:
        scheme, rest = url.split("://", 1)
    else:
        scheme, rest = "", url
    m = re.search(r"[/?#]", rest)
    if m:
        netloc, after = rest[:m.start()], rest[m.start():]
    else:
        netloc, after = rest, ""
    host = netloc.split("@")[-1]
    port = ""
    mp = re.search(r":(\d+)$", host)
    if mp:
        port = mp.group(1)
        host = host[:mp.start()]
    return scheme.lower(), host.lower(), port, after


def entropy_of(text):
    if len(text) == 0:
        return 0.0
    total = 0.0
    for ch in set(text):
        p = text.count(ch) / len(text)
        total -= p * math.log2(p)
    return total


def url_features(url):
    """Return a dict with one number for each name in FEATURE_NAMES."""
    url = str(url).strip()
    n = max(len(url), 1)
    scheme, host, port, after = split_url(url)

    labels = host.split(".")
    is_ip = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host) else 0
    tld = labels[-1] if (len(labels) > 1 and not is_ip) else ""

    path_part = re.split(r"[?#]", after)[0]
    query = after.split("?", 1)[1].split("#")[0] if "?" in after else ""

    letters = sum(1 for c in url if c.isascii() and c.isalpha())
    digits = sum(1 for c in url if c.isdigit())
    other_special = sum(1 for c in url if not c.isalnum() and c not in "=?&")
    digit_runs = re.findall(r"\d+", url)
    lower = url.lower()

    f = {
        "url_length": len(url),
        "domain_length": len(host),
        "is_ip": is_ip,
        "tld_length": len(tld),
        "num_subdomains": 0 if is_ip else max(len(labels) - 2, 0),
        "num_dots_in_host": host.count("."),
        "num_hyphens_in_host": host.count("-"),
        "num_digits_in_host": sum(1 for c in host if c.isdigit()),
        "longest_host_label": max(len(x) for x in labels),
        "num_letters": letters,
        "letter_ratio": letters / n,
        "num_digits": digits,
        "digit_ratio": digits / n,
        "num_equals": url.count("="),
        "num_qmark": url.count("?"),
        "num_ampersand": url.count("&"),
        "num_at": url.count("@"),
        "num_other_special": other_special,
        "special_ratio": (len(url) - letters - digits) / n,
        "num_slashes": url.count("/"),
        "path_length": len(path_part),
        "query_length": len(query),
        "has_query": 1 if query else 0,
        "num_percent": url.count("%"),
        "has_obfuscation": 1 if "%" in url else 0,
        "is_https": 1 if scheme == "https" else 0,
        "has_www": 1 if host.startswith("www.") else 0,
        "has_port": 1 if port else 0,
        "entropy": entropy_of(url),
        "longest_digit_run": max((len(x) for x in digit_runs), default=0),
        "num_suspicious_words": sum(lower.count(w) for w in SUSPICIOUS_WORDS),
        "is_shortener": 1 if host in SHORTENERS else 0,
        "common_tld": 1 if tld in COMMON_TLDS else 0,
    }
    return {k: float(v) for k, v in f.items()}


def features_table(urls):
    """Many URLs -> a DataFrame with the columns in FEATURE_NAMES."""
    rows = [url_features(u) for u in urls]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def strip_scheme(url):
    """Remove 'http://' or 'https://' (used to test how much the scheme matters)."""
    return re.sub(r"^[a-zA-Z]+://", "", str(url))


# ---------- scoring with the saved bundle ----------
# bundle = {"use": "features" | "tfidf" | "both", "feature_model": ..., "text_model": ...,
#           "legit_median": {...}}

def predict_scores(urls, bundle):
    """Risk score between 0 and 1 for each URL (1 = very likely phishing)."""
    urls = [str(u) for u in urls]
    use = bundle["use"]
    parts = []
    if use in ("features", "both"):
        parts.append(bundle["feature_model"].predict_proba(features_table(urls))[:, 1])
    if use in ("tfidf", "both"):
        parts.append(bundle["text_model"].predict_proba(urls)[:, 1])
    return np.mean(parts, axis=0)


def explain_features(url, bundle):
    """What-if table: set one feature at a time to the typical legitimate value and
    see how much the feature model's score drops. A big drop means that feature
    pushed the score up. (An approximation, one feature at a time.)"""
    model = bundle["feature_model"]
    x = features_table([url])
    base = model.predict_proba(x)[0, 1]
    x_many = pd.concat([x] * len(FEATURE_NAMES), ignore_index=True)
    for i, name in enumerate(FEATURE_NAMES):
        x_many.loc[i, name] = bundle["legit_median"][name]
    changed = model.predict_proba(x_many)[:, 1]
    out = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "meaning": [FEATURE_TEXT[n] for n in FEATURE_NAMES],
        "value": [round(float(x.loc[0, n]), 3) for n in FEATURE_NAMES],
        "typical_legit_value": [round(float(bundle["legit_median"][n]), 3) for n in FEATURE_NAMES],
        "effect_on_score": base - changed,
    })
    return out.sort_values("effect_on_score", ascending=False).reset_index(drop=True)


def explain_text(url, bundle, top_n=5):
    """Character pieces of the URL that push the text model towards 'phishing'."""
    pipe = bundle["text_model"]
    vec = pipe.named_steps["tfidf"]
    lr = pipe.named_steps["lr"]
    x = vec.transform([url]).tocoo()
    names = vec.get_feature_names_out()
    contrib = x.data * lr.coef_[0][x.col]
    order = np.argsort(-contrib)[:top_n]
    return [(names[x.col[i]], float(contrib[i])) for i in order if contrib[i] > 0]
