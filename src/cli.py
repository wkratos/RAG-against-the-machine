import fire
from .indexing.indexer import index_corpus
from .retrieval.retriever import search
from .retrieval.dataset_search import search_dataset
from .generation.generator import answer_question


def main() -> None:
    """Command Line Interface"""
    fire.Fire({
        "index": index_corpus,
        "search": search,
        "search_dataset": search_dataset,
        "answer": answer_question
    })
