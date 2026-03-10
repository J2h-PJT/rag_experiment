# ingestion/quality/filters/length_filter.py

class LengthFilter:

    def __init__(self, min_tokens=20):
        self.min_tokens = min_tokens

    def check(self, chunk):

        tokens = chunk.split()

        return len(tokens) >= self.min_tokens