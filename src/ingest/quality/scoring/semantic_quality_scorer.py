# ingestion/quality/scoring/semantic_quality_scorer.py

class SemanticQualityScorer:

    def score(self, chunk):

        words = chunk.split()

        if len(words) < 10:
            return 0.0

        # 간단한 heuristic
        score = min(len(words) / 100, 1.0)

        return score