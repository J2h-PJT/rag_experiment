from typing import List, Tuple, Dict
from src.core.models import Chunk
from src.core.interfaces import IRetriever, ILLMGenerator
from src.qrels.hybrid_retriever import HybridRetriever
from src.qrels.filters import CandidateFilter
from src.qrels.reranker import CrossEncoderReranker
from src.qrels.llm_suggester import LLMSuggester

class GoldQrelsEngine:
    """
    단순 Top-K 추천이 아닌, "IR Evaluation Dataset Builder" (재현성 및 품질 보장)
    8단계 파이프라인 (Retrieval -> RRF -> Filter -> Rerank -> LLM Suggest) 오케스트레이터.
    """
    def __init__(
        self,
        retriever: IRetriever,
        filter_chain: CandidateFilter = None,
        reranker: CrossEncoderReranker = None,
        suggester: LLMSuggester = None
    ):
        self.retriever = retriever
        self.candidate_filter = filter_chain or CandidateFilter()
        self.reranker = reranker or CrossEncoderReranker()
        self.llm_suggester = suggester

    def generate_candidates(
        self, 
        tenant_id: str, 
        query: str, 
        expected_answer: str = "", # Hybrid RRF에선 query+answer 두 Retriever를 묶어서 주입가능
        top_k: int = 20
    ) -> List[Dict]:
        """
        [1~6단계 파이프라인 수행]
        Returns:
            List[Dict]: [{'chunk': Chunk, 'retriever_score': float, 'rerank_score': float, 'llm_suggestion': int}]
        """
        
        # 1. Candidate Generation + 2. Hybrid Retrieval (RRF Fusion)
        # RRF, Noise 처리를 위해 넓은 풀(top_k * 3) 추출
        fetch_k = top_k * 3
        # retriever가 HybridRetriever일 경우 내부적으로 각 서브 모델 동작 & RRF 수행
        retrieved_candidates = self.retriever.retrieve_chunks(tenant_id, query, top_k=fetch_k)
        
        # 3. Candidate Expansion & 4. Candidate Filtering (Duplicate/Noise)
        filtered_candidates = self.candidate_filter.filter_candidates(retrieved_candidates)
        
        # 5. Candidate Diversity & Rerank (CrossEncoder)
        reranked_candidates = self.reranker.rerank(query, filtered_candidates, top_k=top_k)
        
        # 6. LLM Relevance Suggestion (0, 1, 2 점 부여)
        final_list = []
        
        if self.llm_suggester and reranked_candidates:
            # 배치 처리를 위해 청크 분리
            chunks_to_suggest = [c for c, _ in reranked_candidates]
            llm_scores = self.llm_suggester.suggest_scores(query, chunks_to_suggest)
        else:
            llm_scores = {}
            
        # 결과 객체 어셈블리
        for chunk, rerank_score in reranked_candidates:
            final_list.append({
                "chunk": chunk,
                "retriever_score": next((s for c, s in retrieved_candidates if c.chunk_id == chunk.chunk_id), 0.0),
                "rerank_score": rerank_score,
                "llm_suggestion": llm_scores.get(chunk.chunk_id, 0)
            })

        return final_list
