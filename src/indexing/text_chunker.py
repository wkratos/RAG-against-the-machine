def chunk_text(content: str,
               max_chunk_size: int = 2000
               ) -> list[tuple[int, int]]:
    """Split text into chunks and return their character ranges"""
    start = 0
    chunks = []
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be positive")
    while start < len(content):
        end = min(start + max_chunk_size, len(content))
        chunks.append((start, end))
        start = end
    return chunks
