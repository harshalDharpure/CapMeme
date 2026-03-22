import requests
import re
import numpy as np

CONCEPTNET_BASE = "https://api.conceptnet.io"
LIMIT = 20

def _normalize_for_api(text):
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^\w_]", "", text)
    return text[:50] if text else ""

def fetch_concept_labels(hindi_text, lang="hi", limit=LIMIT):
    tokens = hindi_text.split()
    all_labels = []
    seen = set()
    for t in tokens[:10]:
        norm = _normalize_for_api(t)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        try:
            url = f"{CONCEPTNET_BASE}/c/{lang}/{norm}?limit={limit}"
            r = requests.get(url, timeout=3)
            if r.status_code != 200:
                continue
            data = r.json()
            for edge in data.get("edges", [])[:limit]:
                end = edge.get("end", {})
                lab = end.get("label")
                if lab and lab not in seen:
                    all_labels.append(lab)
                    seen.add(lab)
                start = edge.get("start", {})
                lab_s = start.get("label")
                if lab_s and lab_s not in seen:
                    all_labels.append(lab_s)
                    seen.add(lab_s)
        except Exception:
            continue
    return all_labels

def concept_labels_to_embedding(labels, embed_dim, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    if not labels:
        return np.zeros(embed_dim, dtype=np.float32)
    vec = np.zeros(embed_dim, dtype=np.float32)
    for lab in labels:
        h = hash(lab) % (2**30)
        rng = np.random.RandomState(h)
        vec += rng.randn(embed_dim).astype(np.float32)
    vec /= max(len(labels), 1)
    return (vec / (np.linalg.norm(vec) + 1e-8)).astype(np.float32)

def get_kg_embedding(hindi_text, embed_dim=256):
    labels = fetch_concept_labels(hindi_text)
    return concept_labels_to_embedding(labels, embed_dim)
