"""
VeriCheck ML Model Helper — Speed-Optimized v3
===============================================
Key optimizations vs original (10-15s):
  1. Pre-cached TF-IDF vectors for ALL corpus docs at startup (O(1) lookup per comparison)
  2. TF-IDF filter: top-30 candidates per sentence via cosine, not brute-force over 314 corpus
  3. LSTM called only on filtered candidates (max 30 calls per sentence, not 300+)
  4. O(1) IDF lookup via _cached_tfidf_idx_to_token instead of O(vocab) list.search per token
  5. TF-IDF word-level vector pre-computed at startup (no per-request vocab scanning)
  6. LSTM predictions run without verbose flag (eliminates print overhead)
Target latency: < 500ms per request on CPU (was 10-15s)
"""

import os, re, string, pickle, math, time as _time
import difflib

# ─── TensorFlow ────────────────────────────────────────────────────────────
try:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    pad_sequences = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "siamese_lstm_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pickle")
# data.csv lives 3 levels up from ml/ → Capstone 2/
_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))), "data.csv")

# ─── Global caches (built ONCE at startup) ──────────────────────────────────
model = None
tokenizer = None

# Pre-cached corpus
_cached_corpus_texts = []      # list[str]: raw texts in same order as vectors
_cached_corpus_clean = []     # list[str]: pre-cleaned texts

# Pre-cached TF-IDF for EVERY corpus doc (the key optimization)
# Each entry: {token_idx: tfidf_float}
_cached_tfidf_vectors = {}    # dict[int, dict[int,float]]

# TF-IDF model params
_cached_tfidf_vocab = {}        # token(str) → idx(int)
_cached_tfidf_idf = {}         # token(str) → idf(float)
_cached_tfidf_idx_to_token = {} # idx(int) → token(str)  ← O(1) IDF lookup
_cached_tfidf_vocab_size = 0

# How many corpus entries to filter to per sentence
TOP_K = 30          # TF-IDF filter keeps top 30 candidates
LSTM_LIMIT = 2      # LSTM on only top 2 TF-IDF matches → ~4 LSTM calls total for ~5 sentences

# ─── Text Preprocessing ────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lowercase + strip punctuation + normalize whitespace."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

def split_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

# ─── Pre-cache ALL corpus TF-IDF vectors at startup (one-time) ─────────────
def _build_tfidf_caches(corpus: list[str]):
    """Called once at startup. Pre-computes all corpus TF-IDF vectors + vocab."""
    global _cached_tfidf_vocab, _cached_tfidf_idf, _cached_tfidf_idx_to_token
    global _cached_tfidf_vocab_size, _cached_corpus_clean

    _cached_corpus_texts[:] = corpus
    _cached_corpus_clean = [clean_text(t) for t in corpus]

    # Document frequency counts
    doc_count = {}
    tokenized = []
    for doc in corpus:
        tokens = clean_text(doc).split()
        tokenized.append(tokens)
        seen = set()
        for t in tokens:
            if t not in seen:
                seen.add(t)
                doc_count[t] = doc_count.get(t, 0) + 1

    n = len(corpus)

    # Build vocab: top-5000 by document frequency
    freq_sorted = sorted(doc_count.items(), key=lambda x: -x[1])[:5000]
    _cached_tfidf_vocab.clear()
    _cached_tfidf_idf.clear()
    _cached_tfidf_idx_to_token.clear()

    for t, c in freq_sorted:
        idx = len(_cached_tfidf_vocab)
        _cached_tfidf_vocab[t] = idx
        _cached_tfidf_idf[t] = math.log(n / (c + 1)) + 1
        _cached_tfidf_idx_to_token[idx] = t

    _cached_tfidf_vocab_size = len(_cached_tfidf_vocab)

    # Pre-compute TF-IDF vector for every corpus doc (store as dict of {idx: tfidf})
    for i, tokens in enumerate(tokenized):
        tf = {}
        for t in tokens:
            if t in _cached_tfidf_vocab:
                idx = _cached_tfidf_vocab[t]
                tf[idx] = tf.get(idx, 0) + 1
        vec = {}
        for idx, cnt in tf.items():
            vec[idx] = cnt * _cached_tfidf_idf[_cached_tfidf_idx_to_token[idx]]
        _cached_tfidf_vectors[i] = vec

    print(f"  ✓ TF-IDF: vocab={_cached_tfidf_vocab_size}, corpus_vectors={len(_cached_tfidf_vectors)}")


def _tfidf_transform(text: str) -> dict:
    """Return TF-IDF vector for text as dict {idx: score}. O(tokens_in_text)."""
    tokens = clean_text(text).split()
    tf = {}
    for t in tokens:
        if t in _cached_tfidf_vocab:
            idx = _cached_tfidf_vocab[t]
            tf[idx] = tf.get(idx, 0) + 1
    vec = {}
    for idx, cnt in tf.items():
        vec[idx] = cnt * _cached_tfidf_idf[_cached_tfidf_idx_to_token[idx]]
    return vec


def _tfidf_cosine(vec_a: dict, vec_b: dict) -> float:
    """Cosine similarity between two TF-IDF vectors. O(min(nz_a, nz_b))."""
    if not vec_a or not vec_b:
        return 0.0
    common = {k for k in vec_a if k in vec_b}
    if not common:
        return 0.0
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    return dot / (norm_a * norm_b + 1e-10)


def _filter_top_k(text: str, k: int) -> list[tuple]:
    """Return top-k corpus indices by TF-IDF cosine to text. O(V + k*logV)."""
    text_vec = _tfidf_transform(text)
    if not text_vec:
        return []
    scores = []
    for idx, corp_vec in _cached_tfidf_vectors.items():
        sim = _tfidf_cosine(text_vec, corp_vec)
        scores.append((idx, sim))
    # partial sort: nlargest is faster than full sort
    from heapq import nlargest
    return nlargest(k, scores, key=lambda x: x[1])


# ─── Traditional similarity methods ─────────────────────────────────────────
def ngram_similarity(text_a: str, text_b: str, n: int = 2) -> float:
    def get_ngrams(t, n):
        tokens = t.split()
        return set(" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    ga = get_ngrams(text_a, n)
    gb = get_ngrams(text_b, n)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)

def char_ngram_sim(text_a: str, text_b: str, n: int = 4) -> float:
    def char_ngrams(t, n):
        return set(t[i:i+n] for i in range(len(t)-n+1))
    ca = char_ngrams(text_a.lower(), n)
    cb = char_ngrams(text_b.lower(), n)
    if not ca or not cb:
        return 0.0
    return len(ca & cb) / len(ca | cb)

def lcs_ratio(text_a: str, text_b: str) -> float:
    a, b = text_a.lower(), text_b.lower()
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0.0
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            curr[j] = prev[j-1] + 1 if a[i-1] == b[j-1] else max(prev[j], curr[j-1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n] / max(m, n)

def _seqmatch_ratio(a: str, b: str) -> float:
    return float(difflib.SequenceMatcher(None, a, b).ratio())


# ─── Fast ensemble (no LSTM) ───────────────────────────────────────────────
def _fast_similarity(clean_a: str, clean_b: str) -> dict:
    """TF-IDF (O(1) cache) + N-gram + char + LCS. No model.predict()."""
    # O(1) via pre-cached corpus vectors
    vec_a = _tfidf_transform(clean_a)
    vec_b = _tfidf_transform(clean_b)
    tfidf = _tfidf_cosine(vec_a, vec_b)
    bg = ngram_similarity(clean_a, clean_b, n=2)
    tg = ngram_similarity(clean_a, clean_b, n=3)
    ch = char_ngram_sim(clean_a, clean_b, n=4)
    lc = lcs_ratio(clean_a, clean_b)
    sq = _seqmatch_ratio(clean_a, clean_b)
    # Weighted average (no LSTM here)
    ensemble = 0.30 * tfidf + 0.22 * bg + 0.16 * tg + 0.18 * ch + 0.14 * lc
    return {"tfidf": tfidf, "bigram": bg, "trigram": tg,
            "char_ngram": ch, "lcs": lc, "seqmatch": sq,
            "ensemble": min(ensemble, 1.0)}


def _lstm_score(clean_a: str, clean_b: str) -> float:
    """Single LSTM inference. ~50ms per call."""
    if model is None or tokenizer is None or not TF_AVAILABLE:
        return 0.0
    try:
        seq_a = tokenizer.texts_to_sequences([clean_a])
        seq_b = tokenizer.texts_to_sequences([clean_b])
        pad_a = pad_sequences(seq_a, maxlen=50)
        pad_b = pad_sequences(seq_b, maxlen=50)
        return float(model.predict([pad_a, pad_b], verbose=0)[0][0])
    except Exception:
        return 0.0


# ─── Per-sentence analysis (optimized) ─────────────────────────────────────
def _analyze_sentences(text: str) -> tuple:
    """
    Optimized: TF-IDF filter → top-30 candidates → fast ensemble on those →
    LSTM on top-20 → combined score.
    Before: O(sentences × corpus_size × all_methods)  — 10+ seconds
    After:  O(sentences × (V + top_k × fast_methods + top_lstm × lstm))  — <200ms
    """
    sentences = split_sentences(text)
    results = []
    scores = []

    for sent in sentences:
        if len(sent.split()) < 5:
            results.append({"sentence": sent, "score": 0.0, "level": "safe",
                             "match": None, "reason": "Too short to analyze"})
            scores.append(0.0)
            continue

        clean_sent = clean_text(sent)

        # Step 1: TF-IDF filter → top 30 corpus candidates in O(V)
        candidates = _filter_top_k(sent, TOP_K)
        if not candidates:
            results.append({"sentence": sent, "score": 0.0, "level": "safe",
                             "match": None, "reason": "No corpus match"})
            scores.append(0.0)
            continue

        # Step 2: fast ensemble + LSTM on top-20 candidates
        best_score = 0.0
        best_match = None
        best_lstm = 0.0

        for corp_idx, _ in candidates[:LSTM_LIMIT]:
            ref_clean = _cached_corpus_clean[corp_idx]

            # Fast ensemble (TF-IDF + N-gram + char + LCS) — ~0.3ms each
            ens = _fast_similarity(clean_sent, ref_clean)["ensemble"]

            # LSTM on remaining candidates — ~50ms each, max 20 per sentence
            lstm_sc = _lstm_score(clean_sent, ref_clean)
            combined = 0.50 * lstm_sc + 0.50 * ens

            if combined > best_score:
                best_score = combined
                best_lstm = lstm_sc
                best_match = (_cached_corpus_texts[corp_idx][:80] + "...") \
                    if len(_cached_corpus_texts[corp_idx]) > 80 \
                    else _cached_corpus_texts[corp_idx]

        if best_score < 0.2:
            level, reason = "safe", "Original content"
        elif best_score < 0.4:
            level, reason = "low", "Minor similarity to reference material"
        elif best_score < 0.6:
            level, reason = "medium", "Moderate similarity detected"
        elif best_score < 0.75:
            level, reason = "high", "High similarity - review suggested"
        else:
            level, reason = "critical", "Very high similarity - likely plagiarized"

        results.append({
            "sentence": sent, "score": round(best_score, 3), "level": level,
            "match": best_match, "reason": reason,
            "lstm_score": round(best_lstm, 4)
        })
        scores.append(best_score)

    return results, scores


# ─── Build baseline corpus from data.csv ─────────────────────────────────
def build_baseline_corpus() -> list:
    """Load diverse texts from data.csv. Runs once at startup."""
    try:
        import pandas as pd
        if not os.path.exists(_CSV_PATH):
            print(f"  ⚠ data.csv not found at {_CSV_PATH}")
            return []
        df = pd.read_csv(_CSV_PATH)
        df.dropna(subset=["source_txt", "plagiarism_txt"], inplace=True)
        df = df[df["source_txt"].str.len() > 60]

        source_sample = df["source_txt"].astype(str).sample(
            n=min(800, len(df)), random_state=42).tolist()
        plag_sample = df["plagiarism_txt"].astype(str).sample(
            n=min(400, len(df)), random_state=42).tolist()

        academic = [
            "the results of this study indicate that",
            "further research is needed to understand",
            "in conclusion the findings suggest",
            "it is important to note that",
            "the data shows that there is a significant",
            "this research contributes to the field by",
            "based on the analysis of the results",
            "the following section discusses the implications",
            "these findings have important implications for",
            "the purpose of this study was to investigate",
            "the literature review suggests that",
            "the methodology used in this study",
            "the limitations of this study include",
            "the statistical analysis reveals that",
            "the hypothesis was tested using",
            "the control group was used to",
            "the experimental design included",
            "the qualitative analysis shows",
            "the quantitative results indicate",
            "the correlation between the variables was",
        ]

        all_texts = source_sample + plag_sample + academic
        seen = set()
        unique = []
        for t in all_texts:
            tc = t.strip()
            if len(tc) > 80 and tc not in seen:
                seen.add(tc)
                unique.append(tc)
        return unique
    except Exception as e:
        print(f"  ⚠ Baseline corpus error: {e}")
        return []


# ─── Load ML assets at startup ───────────────────────────────────────────────
def load_ml_assets():
    global model, tokenizer
    if TF_AVAILABLE:
        try:
            if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
                model = load_model(MODEL_PATH)
                with open(TOKENIZER_PATH, "rb") as f:
                    tokenizer = pickle.load(f)
                print("✓ LSTM model + tokenizer loaded.")
            else:
                print("⚠ Model/tokenizer files not found — running in fallback mode.")
        except Exception as e:
            print(f"⚠ Model load error: {e} — running in fallback mode.")

    print("Building reference corpus...")
    corpus = build_baseline_corpus()
    print(f"  → {len(corpus)} texts loaded.")

    print("Building TF-IDF caches (one-time startup)...")
    _build_tfidf_caches(corpus)
    print("✓ All caches ready.")


# ─── Main entry point ────────────────────────────────────────────────────────
def detect_plagiarism(text: str) -> dict:
    """
    Full plagiarism detection pipeline.
    Target: < 500ms on CPU for any input size.
    """
    t0 = _time.time()
    word_count = len(text.split())
    char_count = len(text)

    if word_count < 3:
        return {
            "similarity": 0.0, "level": "safe",
            "highlighted_html": f"<p>{text}</p>",
            "sentence_analysis": [], "metrics": {},
            "word_count": word_count, "char_count": char_count,
            "explanation": "Text too short to analyze.",
            "latency_ms": 0.0
        }

    # Per-sentence analysis (optimized)
    sentence_data, sentence_scores = _analyze_sentences(text)

    max_sim = max(sentence_scores) if sentence_scores else 0.0
    avg_sim = sum(sentence_scores) / len(sentence_scores) if sentence_scores else 0.0
    lstm_best = max((s.get("lstm_score") or 0) for s in sentence_data)

    # Final blended score
    if lstm_best > 0:
        final_score = 0.50 * lstm_best + 0.30 * max_sim + 0.20 * avg_sim
    else:
        final_score = 0.60 * max_sim + 0.40 * avg_sim
    final_score = min(final_score, 1.0)

    # Risk level
    if final_score < 0.20:
        level, explanation = "safe", "Your text appears to be original with no significant similarity to known sources."
    elif final_score < 0.40:
        level, explanation = "low", "Minor similarity detected. Some phrases may overlap with common academic expressions. This is typically not a concern."
    elif final_score < 0.60:
        level, explanation = "medium", "Moderate similarity found in some sections. Consider revising or citing sources where indicated."
    elif final_score < 0.75:
        level, explanation = "high", "High similarity detected across multiple sections. Review flagged areas and ensure proper citation."
    else:
        level, explanation = "critical", "Very high similarity detected. Significant sections closely match known sources. Immediate revision and citation required."

    # Build highlighted HTML
    color_map = {
        "safe": "#10b981", "low": "#10b981",
        "medium": "#f59e0b", "high": "#ef4444", "critical": "#dc2626"
    }
    highlighted_parts = []
    for item in sentence_data:
        sent = item["sentence"]
        score = item["score"]
        color = color_map.get(item["level"], "#9ca3af")
        esc = sent.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        pct = str(round(score * 100)) + "%"
        reason_esc = item["reason"].replace("&", "&amp;").replace('"', "&quot;")
        if score >= 0.40:
            tag = (f'<span style="background:{color}18;border-left:3px solid {color};'
                   f'padding:2px 6px;margin:2px 0;display:block;border-radius:4px;" '
                   f'title="Risk: {pct} — {reason_esc}">'
                   f'{esc} <span style="font-size:0.75em;color:{color};float:right;">{pct}</span></span>')
            highlighted_parts.append(tag)
        else:
            highlighted_parts.append(f'<span style="color:#9ca3af;">{esc}</span>')

    highlighted_html = "<div class='sent-analysis'>" + " ".join(highlighted_parts) + "</div>"
    elapsed_ms = round((_time.time() - t0) * 1000, 1)

    return {
        "similarity": round(final_score, 4),
        "level": level,
        "highlighted_html": highlighted_html,
        "sentence_analysis": sentence_data,
        "metrics": {
            "lstm_best": round(lstm_best, 4) if lstm_best else None,
            "max_similarity": round(max_sim, 4),
            "avg_similarity": round(avg_sim, 4),
            "sentences_analyzed": len(sentence_data),
            "risky_sentences": sum(1 for s in sentence_scores if s >= 0.4)
        },
        "word_count": word_count,
        "char_count": char_count,
        "explanation": explanation,
        "latency_ms": elapsed_ms
    }


# ─── Backward-compat export for routes.py ─────────────────────────────────
def ensemble_similarity(text_a: str, text_b: str, tfidf_vec=None) -> dict:
    """Exported for routes.py free-check endpoint (two-text comparison)."""
    clean_a = clean_text(text_a)
    clean_b = clean_text(text_b)
    scores = _fast_similarity(clean_a, clean_b)
    lstm_sc = _lstm_score(clean_a, clean_b)
    scores["lstm"] = lstm_sc
    if lstm_sc > 0:
        scores["ensemble"] = 0.35 * lstm_sc + 0.65 * scores["ensemble"]
    return scores


def generate_highlighted_diff(text_a: str, text_b: str) -> str:
    """Highlights sentences in A structurally matching B (for compare endpoint)."""
    sents_a = re.split(r'(?<=[.!?])\s+', text_a)
    sents_b = re.split(r'(?<=[.!?])\s+', text_b)
    highlighted = []
    for s_a in sents_a:
        if not s_a.strip():
            continue
        max_ratio = 0.0
        for s_b in sents_b:
            if not s_b.strip():
                continue
            ratio = difflib.SequenceMatcher(None, s_a.lower(), s_b.lower()).ratio()
            if ratio > max_ratio:
                max_ratio = ratio
        if max_ratio > 0.65:
            esc = s_a.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            highlighted.append(
                f'<mark style="background:rgba(239,68,68,0.18);border-bottom:2px solid #ef4444;'
                f'padding:1px 3px;border-radius:3px;">{esc}</mark>'
            )
        else:
            highlighted.append(s_a)
    return " ".join(highlighted)