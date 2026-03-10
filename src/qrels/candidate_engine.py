from typing import List, Tuple, Dict
from src.core.models import Chunk
from src.core.interfaces import IDocumentRepository, IEmbedder

class CandidateEngine:
    """
    질문(Query)과 모범 정답(Expected Answer)에 대한 임베딩을 각각 수행하여,
    각 Chunk에 대한 유사도 점수를 다중 계산(Multi-score)한 후 최종 후보군을 도출합니다.
    """
    def __init__(self, doc_repo: IDocumentRepository, embedder: IEmbedder):
        self.doc_repo = doc_repo
        self.embedder = embedder

    def retrieve_candidates(
        self, 
        tenant_id: str, 
        query: str, 
        expected_answer: str, 
        top_k: int = 5,
        query_weight: float = 0.4,   # w1
        answer_weight: float = 0.6   # w2
    ) -> List[Dict]:
        """
        Query 임베딩과 Answer 임베딩을 이용해 검색하고,
        두 Distance 값을 종합 계산하여 최종 Top K를 반환합니다.
        
        Returns:
            List[Dict]: [{'chunk': Chunk, 'score': float, 'query_distance': float, 'answer_distance': float}]
        """
        # 1. 질의와 정답의 벡터 임베딩 추출
        query_emb = self.embedder.embed_text(query)
        answer_emb = self.embedder.embed_text(expected_answer)
        
        # 2. 풀을 넓게 잡아 각 조건별 후보 검색 (top_k * 3배수 정도 추출 후 병합)
        fetch_k = top_k * 3
        query_results = self.doc_repo.search_chunks_by_embedding(tenant_id, query_emb, top_k=fetch_k)
        answer_results = self.doc_repo.search_chunks_by_embedding(tenant_id, answer_emb, top_k=fetch_k)
        
        # 3. 결과 합치기 (Chunk ID 기준)
        # distance는 낮을수록 좋음 (pgvector `<=>`는 0에 가까울수록 매칭)
        # 기본값으로 못 찾은 경우 distance를 1.0 (또는 적절한 패널티값)으로 줌
        chunk_map = {}
        
        for chunk, dist in query_results:
            chunk_map[chunk.chunk_id] = {
                "chunk": chunk,
                "query_distance": dist,
                "answer_distance": 1.0  # 초기값
            }
            
        for chunk, dist in answer_results:
            if chunk.chunk_id in chunk_map:
                chunk_map[chunk.chunk_id]["answer_distance"] = dist
            else:
                chunk_map[chunk.chunk_id] = {
                    "chunk": chunk,
                    "query_distance": 1.0, # 검색 안 된 경우 패널티
                    "answer_distance": dist
                }
                
        # 4. Multi-score 계산 (Distance 결합)
        # Distance가 낮을수록 좋으므로, 가중합 Distance가 오름차순이 되도록 정렬
        final_candidates = []
        for v in chunk_map.values():
            combined_distance = (v["query_distance"] * query_weight) + (v["answer_distance"] * answer_weight)
            # 직관성을 위해 score를 (1 - distance) 로 역산정의 (1.0에 가까울수록 좋음)
            similarity_score = 1.0 - (combined_distance / 2.0)
            
            final_candidates.append({
                "chunk": v["chunk"],
                "score": round(similarity_score, 4),
                "query_distance": round(v["query_distance"], 4),
                "answer_distance": round(v["answer_distance"], 4)
            })
            
        # 5. 최종 Score 기준 내림차순 정렬 (높은 Score가 상단)
        final_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        return final_candidates[:top_k]
