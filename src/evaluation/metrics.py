import math
from typing import List, Dict


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Recall@K: 상위 K개 중 관련 문서 비율"""
    if not relevant_ids:
        return 0.0
    retrieved_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(retrieved_k & relevant) / len(relevant)


def mrr_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """MRR@K: 첫 번째 관련 문서의 역순위 평균"""
    relevant = set(relevant_ids)
    for i, chunk_id in enumerate(retrieved_ids[:k]):
        if chunk_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def _dcg(scores: List[float], k: int) -> float:
    return sum(score / math.log2(i + 2) for i, score in enumerate(scores[:k]))


def ndcg_at_k(retrieved_ids: List[str], qrel_scores: Dict[str, int], k: int) -> float:
    """NDCG@K: 정규화 할인 누적 이득 (operator_score 기반)"""
    actual_scores = [qrel_scores.get(cid, 0) for cid in retrieved_ids[:k]]
    ideal_scores  = sorted(qrel_scores.values(), reverse=True)
    ideal_dcg     = _dcg(ideal_scores, k)
    if ideal_dcg == 0:
        return 0.0
    return _dcg(actual_scores, k) / ideal_dcg


def compute_metrics(retrieved_ids: List[str], qrel_scores: Dict[str, int], k: int) -> Dict[str, float]:
    """단일 질문에 대한 전체 메트릭 계산"""
    relevant_ids = [cid for cid, score in qrel_scores.items() if score >= 1]
    return {
        f"Recall@{k}": recall_at_k(retrieved_ids, relevant_ids, k),
        f"MRR@{k}":    mrr_at_k(retrieved_ids, relevant_ids, k),
        f"NDCG@{k}":   ndcg_at_k(retrieved_ids, qrel_scores, k),
    }
