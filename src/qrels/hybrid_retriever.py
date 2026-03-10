from typing import List, Tuple, Dict
from src.core.models import Chunk
from src.core.interfaces import IRetriever

class HybridRetriever(IRetriever):
    """
    다중 Retriever(Vector, BM25 등)의 결과를 취합하여
    가중치 적용 RRF(Reciprocal Rank Fusion) 공식으로 순위를 통합합니다.

    weighted RRF score = weight_i / (k + rank_i)
    """
    def __init__(self, retrievers: List[IRetriever], weights: List[float] = None, k: int = 60):
        """
        Args:
            retrievers: 병합할 검색 엔진 목록 (예: [VectorRetriever, BM25Retriever])
            weights:    각 retriever의 가중치 (None이면 균등 분배). 합이 1.0이 되도록 정규화됨.
            k:          RRF 상수 (보통 60 사용)
        """
        self.retrievers = retrievers
        self.k = k

        if weights is None:
            n = len(retrievers)
            self.weights = [1.0 / n] * n
        else:
            total = sum(weights) or 1.0
            self.weights = [w / total for w in weights]  # 정규화

    def retrieve_chunks(self, tenant_id: str, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        fetch_k = top_k * 2

        all_results: List[List[Tuple[Chunk, float]]] = []
        for retriever in self.retrievers:
            results = retriever.retrieve_chunks(tenant_id, query, top_k=fetch_k)
            all_results.append(results)

        # 가중치 적용 RRF 병합
        rrf_map: Dict[str, Tuple[Chunk, float]] = {}
        for search_results, weight in zip(all_results, self.weights):
            for rank, (chunk, _) in enumerate(search_results, start=1):
                rrf_score = weight / (self.k + rank)
                if chunk.chunk_id in rrf_map:
                    existing, cur_score = rrf_map[chunk.chunk_id]
                    rrf_map[chunk.chunk_id] = (existing, cur_score + rrf_score)
                else:
                    rrf_map[chunk.chunk_id] = (chunk, rrf_score)

        sorted_results = sorted(rrf_map.values(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
