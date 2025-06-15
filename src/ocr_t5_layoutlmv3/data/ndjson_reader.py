import ijson

def wrap_json_lines(file):
    """Wrap NDJSON lines into a valid JSON array."""
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
    """File-like reader for iterator-based JSON parsing."""
    def __init__(self, iterator):
        self.iterator = iterator
        self.buffer = ''

    def read(self, size=-1):
        if size < 0:
            return ''.join(self.iterator)
        while len(self.buffer) < size:
            try:
                self.buffer += next(self.iterator)
            except StopIteration:
                break
        result = self.buffer[:size]
        self.buffer = self.buffer[size:]
        return result

def iter_ndjson_in_chunks(json_path, chunk_size=1000):
    """Iterate through NDJSON file in chunks."""
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