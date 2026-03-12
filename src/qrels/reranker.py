from typing import List, Tuple
from src.core.models import Chunk

try:
    from sentence_transformers import CrossEncoder
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


def _normalize(scores: List[float]) -> List[float]:
    """Min-Max 정규화 → [0, 1] 범위로 변환"""
    if not scores:
        return scores
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [0.5] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


class CrossEncoderReranker:
    """
    Candidate를 Cross-Encoder 점수로 재정렬합니다.

    fusion_weight 파라미터로 Retrieval 점수와 Reranker 점수를 혼합합니다:
      fusion_weight=1.0 → 순수 Reranker 점수만 사용 (기존 방식)
      fusion_weight=0.5 → Retrieval 50% + Reranker 50% 혼합
      fusion_weight=0.0 → Retrieval 점수만 사용 (Reranker 미적용과 동일)
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        if HAS_SENTENCE_TRANSFORMERS:
            self.model = CrossEncoder(model_name)
        else:
            print(f"[Warning] sentence-transformers 미설치. Dummy reranking 적용: {model_name}")

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Chunk, float]],
        top_k: int = 20,
        fusion_weight: float = 1.0,
    ) -> List[Tuple[Chunk, float]]:
        """
        fusion_weight: Reranker 점수 비중 (0.0~1.0)
          - 1.0: 순수 Reranker (기존 동작)
          - 0.5: Retrieval + Reranker 혼합
          - 0.0: Retrieval 점수 그대로 (Reranker 건너뜀)
        """
        if not candidates:
            return []

        if not HAS_SENTENCE_TRANSFORMERS or self.model is None or fusion_weight == 0.0:
            return candidates[:top_k]

        # CrossEncoder 점수 계산
        pairs = [[query, chunk.text] for chunk, _ in candidates]
        reranker_scores = list(self.model.predict(pairs))

        if fusion_weight >= 1.0:
            # 순수 Reranker — 기존 방식
            result = [
                (chunk, float(reranker_scores[i]))
                for i, (chunk, _) in enumerate(candidates)
            ]
        else:
            # Score Fusion: 두 점수를 각각 정규화 후 가중 혼합
            original_scores = [float(s) for _, s in candidates]
            norm_orig = _normalize(original_scores)
            norm_rank = _normalize(reranker_scores)

            result = []
            for i, (chunk, _) in enumerate(candidates):
                fused = (1.0 - fusion_weight) * norm_orig[i] + fusion_weight * norm_rank[i]
                result.append((chunk, fused))

        result.sort(key=lambda x: x[1], reverse=True)
        return result[:top_k]
