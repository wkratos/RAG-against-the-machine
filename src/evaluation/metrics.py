import json


def source_iou(a: dict, b: dict) -> float:
    """Calculate the overlap between two source ranges."""
    start = max(
        a["first_character_index"],
        b["first_character_index"]
    )
    end = min(
        a["last_character_index"],
        b["last_character_index"]
    )
    intersection = max(0, end - start)
    length_a = a["last_character_index"] - a["first_character_index"]
    length_b = b["last_character_index"] - b["first_character_index"]

    union = length_a + length_b - intersection
    if union <= 0:
        return 0
    return (intersection / union)


def is_match(retrieved: dict, expected: dict) -> bool:
    """Check whether a retrieved source matches a reference."""
    if retrieved["file_path"] != expected["file_path"]:
        return False
    return source_iou(retrieved, expected) >= 0.05


def evaluate_recall(reference_path: str, results_path: str) -> float:
    """Calculate average source recall over all questions."""
    total_recall = 0.0
    count = 0

    with open(reference_path, "r", encoding="utf-8") as reference:
        expected = json.load(reference)
    with open(results_path, "r", encoding="utf-8") as file:
        results = json.load(file)

    results_by_id = {
        result["question_id"]: result
        for result in results["search_results"]
    }
    for question in expected["rag_questions"]:
        question_id = question["question_id"]
        expected_sources = question["sources"]
        if not expected_sources:
            continue
        retrieved_sources = results_by_id.get(
            question_id, {}).get("retrieved_sources", [])

        matched = 0
        for ex in expected_sources:
            if any(is_match(result, ex) for result in retrieved_sources):
                matched += 1
        recall = matched / len(expected_sources)
        total_recall += recall
        count += 1

    return total_recall / count if count > 0 else 0.0


def analyze_failures(reference_path: str, results_path: str) -> None:
    """Report why retrieved sources fail to match references."""
    with open(reference_path, "r", encoding="utf-8") as reference:
        expected = json.load(reference)
    with open(results_path, "r", encoding="utf-8") as file:
        results = json.load(file)

    results_by_id = {
        result["question_id"]: result
        for result in results["search_results"]
    }
    for question in expected["rag_questions"]:
        question_id = question["question_id"]
        expected_sources = question["sources"]
        if not expected_sources:
            continue
        retrieved_sources = results_by_id.get(
            question_id, {}).get("retrieved_sources", [])
        for ex in expected_sources:
            if any(is_match(ret, ex) for ret in retrieved_sources):
                continue
            elif any(ret["file_path"] == ex["file_path"]
                     for ret in retrieved_sources):
                print("WRONG RANGE:", question["question"])
            else:
                print("FILE MISSING:", question["question"])
