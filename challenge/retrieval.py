"""
BM25 + FAISS hybrid retrieval with Reciprocal Rank Fusion (RRF).

Preprocessing:
  - Lowercase
  - Remove English stopwords (NLTK if available, else built-in fallback)
  - Unigrams + bigrams + trigrams

Usage:
  from retrieval import BM25Retriever, hybrid_retrieve
"""

from __future__ import annotations

import re

# -- stopwords -----------------------------------------------------------------

_BUILTIN_STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","being","have","has","had","do","does",
    "did","will","would","could","should","may","might","shall","can","not",
    "no","nor","so","yet","both","either","neither","than","rather","whether",
    "i","you","he","she","it","we","they","me","him","her","us","them",
    "my","your","his","its","our","their","this","that","these","those",
    "who","whom","which","what","where","when","why","how","all","each",
    "every","any","few","more","most","other","some","such","into","through",
    "during","before","after","above","below","between","out","off","over",
    "under","again","then","once","here","there","about","up","down","from",
    "by","as","if","while","also","just","because","until","since","etc",
}

_STOPWORDS: set | None = None


def _get_stopwords() -> set:
    try:
        import nltk
        try:
            from nltk.corpus import stopwords
            return set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            return set(stopwords.words("english"))
    except Exception:
        return _BUILTIN_STOPWORDS


def _stopwords() -> set:
    global _STOPWORDS
    if _STOPWORDS is None:
        _STOPWORDS = _get_stopwords()
    return _STOPWORDS


# -- preprocessing -------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._\-][a-z0-9]+)*")


def preprocess(text: str, max_ngram: int = 3) -> list:
    """
    Lowercase, remove stopwords, return unigrams + bigrams + trigrams.

    "machine learning engineer" ->
        ["machine", "learning", "engineer",
         "machine_learning", "learning_engineer",
         "machine_learning_engineer"]
    """
    sw = _stopwords()
    tokens = [t for t in _TOKEN_RE.findall(text.lower()) if t not in sw]

    result = list(tokens)
    for n in range(2, max_ngram + 1):
        for i in range(len(tokens) - n + 1):
            result.append("_".join(tokens[i : i + n]))

    return result


# -- BM25 retriever ------------------------------------------------------------

class BM25Retriever:
    """
    BM25Okapi over candidate profiles with n-gram preprocessing.

    Usage:
        r = BM25Retriever()
        r.build(profiles)              # list of {"candidate_id", "profile_text"}
        scores = r.score_all(jd_text)  # {cid: float}
        top = r.retrieve(jd_text, top_k=500)
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75, max_ngram: int = 3):
        self.k1 = k1
        self.b = b
        self.max_ngram = max_ngram
        self._bm25 = None
        self._ids: list = []

    def build(self, profiles: list) -> None:
        """Build BM25 index. profiles = [{"candidate_id": str, "profile_text": str}]"""
        from rank_bm25 import BM25Okapi  # type: ignore

        self._ids = [p["candidate_id"] for p in profiles]
        tokenized = [preprocess(p.get("profile_text", ""), self.max_ngram) for p in profiles]
        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        print(f"[bm25] Index built over {len(self._ids):,} documents")

    def score_all(self, jd_text: str) -> dict:
        """Return {candidate_id: raw_bm25_score} for all indexed candidates."""
        if self._bm25 is None:
            raise RuntimeError("Call build() first")
        query_tokens = preprocess(jd_text, self.max_ngram)
        raw_scores = self._bm25.get_scores(query_tokens)
        return {cid: float(s) for cid, s in zip(self._ids, raw_scores)}

    def retrieve(self, jd_text: str, top_k: int = 500) -> list:
        """Return top-k candidate_ids sorted by BM25 score descending."""
        scores = self.score_all(jd_text)
        return sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]


# -- Reciprocal Rank Fusion ----------------------------------------------------

def reciprocal_rank_fusion(ranked_lists: list, k: int = 60) -> list:
    """
    Merge multiple ranked lists via RRF (Cormack et al. 2009).
    RRF score = sum(1 / (k + rank_i)) across all lists.
    Returns merged list sorted by RRF score descending.
    k=60 is the standard default.
    """
    rrf: dict = {}
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked, start=1):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(rrf, key=lambda x: rrf[x], reverse=True)


def hybrid_retrieve(
    bm25_ids: list,
    faiss_ids: list,
    top_k: int,
    rrf_k: int = 60,
) -> list:
    """
    Merge BM25 and FAISS ranked lists via RRF, return top_k candidates.
    """
    merged = reciprocal_rank_fusion([bm25_ids, faiss_ids], k=rrf_k)
    return merged[:top_k]
