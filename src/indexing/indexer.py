from pathlib import Path
from .python_chunker import chunk_python
from .text_chunker import chunk_text
import json


def index_file(
    filepath: str,
    max_chunk_size: int = 2000
) -> list[dict]:
    """Read and chunk one source file."""
    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read()
    if Path(filepath).suffix == ".py":
        chunks = chunk_python(content, max_chunk_size)
    else:
        chunks = chunk_text(content, max_chunk_size)
    records = []
    for start, end in chunks:
        record = {
            "file_path": filepath,
            "first_character_index": start,
            "last_character_index": end,
            "text": content[start:end]
        }
        records.append(record)

    return records

def index_corpus(
    source_directory: str,
    output_directory: str,
    max_chunk_size: int = 2000
) -> None:
    """Index all supported files in a directory."""
    records = []
    for path in Path(source_directory).rglob("*"):
        if (path.is_file()
            and path.suffix.lower() in {".py", ".md", ".txt"}):
            records.extend(index_file(str(path), max_chunk_size))
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "index.json", "w") as file:
        json.dump(records, file)
   
