import joblib
from functools import lru_cache


@lru_cache(maxsize=1)
def load_index():
    """Load the saved TF-IDF index once."""
    return joblib.load("data/processed/vmr.joblib")


def search(query: str, k: int = 5) -> list[dict]:
    """Return the top-k most relevant indexed sources."""
    file_counts = {}
    if k <= 0 or not query.strip():
        return []
    vect, matrix, records = load_index()
    tops = []
    query_vector = vect.transform([query])
    scores = (matrix @ query_vector.T).toarray().ravel()
    for i, record in enumerate(records):
        if record["file_path"].endswith(".md"):
            scores[i] *= 1.5
    ranked_indices = scores.argsort()[::-1]
    for i in ranked_indices:
        record = records[int(i)]
        path = record["file_path"]
        if scores[i] <= 0:
            continue
        if file_counts.get(path, 0) >= 2:
            continue
        file_counts[path] = file_counts.get(path, 0) + 1
        tops.append(record)
        if len(tops) >= k:
            break
    return tops
