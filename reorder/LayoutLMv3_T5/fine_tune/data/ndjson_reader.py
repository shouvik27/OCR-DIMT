"""Module for reading NDJSON files in chunks."""

import ijson
from typing import Iterator, List, Any

def wrap_json_lines(file: Iterator[str]) -> Iterator[str]:
    """Wrap individual JSON lines into a valid JSON array.
    
    Args:
        file: Iterator yielding individual JSON lines
        
    Yields:
        Strings that form a valid JSON array when combined
    """
    yield '['
    first = True
    for line in file:
        line = line.strip()
        if not line:
            continue
        if not first:
            yield ','
        first = False
        yield line
    yield ']'

class IteratorReader:
    """File-like object that reads from an iterator."""
    
    def __init__(self, iterator: Iterator[str]):
        """Initialize with an iterator."""
        self.iterator = iterator
        self.buffer = ''

    def read(self, size: int = -1) -> str:
        """Read up to size bytes from the iterator.
        
        Args:
            size: Number of bytes to read (-1 for all available)
            
        Returns:
            String containing the read data
        """
        if size < 0:
            return ''.join(list(self.iterator))
        while len(self.buffer) < size:
            try:
                self.buffer += next(self.iterator)
            except StopIteration:
                break
        result = self.buffer[:size]
        self.buffer = self.buffer[size:]
        return result

def iter_ndjson_in_chunks(json_path: str, chunk_size: int = 1000) -> Iterator[List[Any]]:
    """Iterate through an NDJSON file in chunks.
    
    Args:
        json_path: Path to the NDJSON file
        chunk_size: Number of items per chunk
        
    Yields:
        Lists of parsed JSON objects (chunks)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        wrapped_iter = wrap_json_lines(f)
        reader = IteratorReader(wrapped_iter)
        objects = ijson.items(reader, 'item')
        chunk = []
        for obj in objects:
            chunk.append(obj)
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk