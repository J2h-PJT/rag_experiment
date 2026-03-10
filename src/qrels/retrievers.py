from typing import List, Tuple
from src.core.models import Chunk
from src.core.interfaces import IRetriever, IDocumentRepository, IEmbedder

class VectorRetriever(IRetriever):
    """
    pgvector 기반 Cosine Distance 벡터 검색기
    """
    def __init__(self, doc_repo: IDocumentRepository, embedder: IEmbedder):
        self.doc_repo = doc_repo
        self.embedder = embedder

    def retrieve_chunks(self, tenant_id: str, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        # 1. 쿼리를 임베딩하여 벡터 변환
        query_emb = self.embedder.embed_text(query)

        # 2. DB (pgvector) 질의 — 동일 embedding_model 청크만 검색
        results = self.doc_repo.search_chunks_by_embedding(
            tenant_id, query_emb, self.embedder.get_model_name(), top_k=top_k
        )
        
        # 3. Hybrid Fusion을 위해 직관적인 Score(1-distance 등)로 반환하거나 원본 그대로 반환
        # RRF(Reciprocal Rank Fusion)의 경우 distance나 score의 절대값보다는 랭킹이 중요하므로
        # distance 오름차순(점수가 좋을수록 상위)으로 정렬된 현재 상태를 유지.
        
        return results

from rank_bm25 import BM25Okapi

class BM25Retriever(IRetriever):
    """
    rank-bm25 기반 Keyword 검색기.
    실제 운영 환경에선 ElasticSearch/Opensearch 등을 쓰지만, 
    연구용 플랫폼에선 로컬 인덱싱 방식으로도 충분함.
    """
    def __init__(self, doc_repo: IDocumentRepository, tokenizer_fn=None):
        self.doc_repo = doc_repo
        self.tokenizer = tokenizer_fn or (lambda x: x.split())
        self.index = None
        self.chunks_cache: List[Chunk] = []

    def _ensure_index(self, tenant_id: str):
        """특정 테넌트의 모든 청크를 가져와 BM25 인덱스를 생성합니다 (Lazy Loading)"""
        if self.index is not None:
            return
            
        # 1. 모든 청크 로드
        self.chunks_cache = self.doc_repo.list_all_chunks_by_tenant(tenant_id)
        if not self.chunks_cache:
            return

        # 2. 토큰화 및 BM25 인덱싱
        tokenized_corpus = [self.tokenizer(c.text) for c in self.chunks_cache]
        self.index = BM25Okapi(tokenized_corpus)

    def retrieve_chunks(self, tenant_id: str, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        self._ensure_index(tenant_id)
        if not self.index or not self.chunks_cache:
            return []
            
        tokenized_query = self.tokenizer(query)
        # BM25 점수 계산
        scores = self.index.get_scores(tokenized_query)
        
        # 상위 K개 추출
        import numpy as np
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for i in top_indices:
            if scores[i] > 0: # 0점 초과인 것들만 (검색어 포함된 것들)
                # BM25 점수를 Cosine Distance와 유사한 척도로 변환하긴 어려우나,
                # 일단 점수 그대로 반환
                results.append((self.chunks_cache[i], float(scores[i])))
        
        return results
