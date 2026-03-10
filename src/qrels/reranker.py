from typing import List, Tuple
from src.core.models import Chunk

# 테스트 용이나 시스템 미비시를 위한 Mock 처리. 실제로는 sentence-transformers 패키지가 필요합니다.
try:
    from sentence_transformers import CrossEncoder
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

class CrossEncoderReranker:
    """
    Candidate를 Semantic Ranking으로 재정렬합니다.
    예: cross-encoder/ms-marco-MiniLM-L-6-v2 활용
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        if HAS_SENTENCE_TRANSFORMERS:
            # 첫 호출 시 모델 로드 최적화를 위해 여기서 로드할 수도 있음
            self.model = CrossEncoder(model_name)
        else:
            print(f"[Warning] `sentence-transformers` is not installed. Dummy reranking will be applied for {model_name}.")

    def rerank(self, query: str, candidates: List[Tuple[Chunk, float]], top_k: int = 20) -> List[Tuple[Chunk, float]]:
        """
        Cross-Encoder를 통해 주어진 Query와 Chunk List 간의 Semantic score를
        측정하고, 상위 top_k개만 내림차순(점수 높은 순)으로 반환합니다.
        """
        if not candidates:
            return []

        # sentence-transformers가 없는 경우(테스트 등), 그냥 원본 리스트 자르기 반환
        if not HAS_SENTENCE_TRANSFORMERS or self.model is None:
            return candidates[:top_k]
            
        # CrossEncoder 형식에 맞게 Pair 구성 (Query, Document_Text)
        pairs = [[query, chunk.text] for chunk, _ in candidates]
        
        # 모델 예측 (Scores 반환)
        scores = self.model.predict(pairs)
        
        # 원본 청크와 점수를 매핑하여 정렬
        reranked = []
        for i, (chunk, _) in enumerate(candidates):
            reranked.append((chunk, float(scores[i])))
            
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]
