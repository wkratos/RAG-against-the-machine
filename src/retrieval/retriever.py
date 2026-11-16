from sklearn.feature_extraction.text import TfidfVectorizer
import json
from pathlib import Path


def search(query: str, k: int = 5) -> list[dict]:
    """Return the top-k most relevant indexed sources."""
    if k <= 0 or not query.strip():
        return []
    records = json.loads(
    Path("data/processed/index.json").read_text()
    )
    documents = [record["text"] for record in records]

    tops = []
    vect = TfidfVectorizer()
    matrix = vect.fit_transform(documents)
    query_vector = vect.transform([query])
    scores = (matrix @ query_vector.T).toarray().ravel()
    ranked_indices = scores.argsort()[::-1]
    for i in ranked_indices[:k]:
        tops.append(records[int(i)])
    return tops

