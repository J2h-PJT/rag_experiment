# ingestion/quality/chunk_quality_engine.py

class ChunkQualityEngine:

    def __init__(
        self,
        filters,
        scorer=None,
        score_threshold=0.2
    ):
        self.filters = filters
        self.scorer = scorer
        self.score_threshold = score_threshold

    def process(self, chunks):

        filtered_chunks = []

        for chunk in chunks:

            if not self._pass_filters(chunk):
                continue

            if self.scorer:

                score = self.scorer.score(chunk)

                if score < self.score_threshold:
                    continue

            filtered_chunks.append(chunk)

        return filtered_chunks

    def _pass_filters(self, chunk):

        for f in self.filters:

            if not f.check(chunk):
                return False

        return True