import joblib
from functools import lru_cache


@lru_cache(maxsize=1)
def load_index():
    """Load the saved TF-IDF index once."""
    return joblib.load("data/processed/vmr.joblib")


def search(query: str, k: int = 5) -> list[dict]:
    """Return the top-k most relevant indexed sources."""
    if k <= 0 or not query.strip():
        return []
    vect, matrix, records = load_index()
    tops = []
    query_vector = vect.transform([query])
    scores = (matrix @ query_vector.T).toarray().ravel()
    ranked_indices = scores.argsort()[::-1]
    for i in ranked_indices[:k]:
        tops.append(records[int(i)])
    return tops
