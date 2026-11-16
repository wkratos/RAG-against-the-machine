from .retriever import search
from ..models import MinimalSource, MinimalSearchResults, StudentSearchResults
import json
from pathlib import Path

def search_dataset(
    dataset_path: str,
    k: int = 5,
    save_directory: str = "data/output/search_results"
) -> None:
    """Search all questions in a dataset and save the results."""
    with open(dataset_path, "r", encoding="utf-8") as file:
        dataset = json.load(file)
    results = []
    for question in dataset["rag_questions"]:
        sources = []
        retrieved = search(question["question"], k)
        for record in retrieved:
            source = MinimalSource(
                file_path=record["file_path"],
                first_character_index=record["first_character_index"],
                last_character_index=record["last_character_index"]
            )
            sources.append(source)
        result = MinimalSearchResults(
            question_id=question["question_id"],
            question=question["question"],
            retrieved_sources=sources
        )
        results.append(result)
    output = StudentSearchResults(
        search_results=results,
        k=k
    )
    output_path = Path(save_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / Path(dataset_path).name

    file_path.write_text(
        output.model_dump_json(indent=2),
        encoding="utf-8"
    )
