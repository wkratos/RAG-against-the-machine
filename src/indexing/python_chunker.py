import ast


def chunk_python(
        content: str,
        max_chunk_size: int = 2000
        ) -> list[tuple[int, int]]:
    """Split python structures into chunks 
    and return their structures ranges"""
    chunks = []
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size should be positif")
    lines = content.splitlines(keepends=True)
    tree = ast.parse(content)
    for node in tree.body:
        start = sum(
            len(line) for line in lines[:node.lineno - 1])
        end = sum(
            len(line) for line in lines[:node.end_lineno])
        while start < end:
            chunk_end = min(end, start + max_chunk_size)
            chunks.append((start, chunk_end))
            start = chunk_end
    return chunks
