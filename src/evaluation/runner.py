import uuid
import datetime
from typing import Dict, List, Optional, Tuple

from src.core.models import ExperimentRun, RetrievalResult, EvaluationResult, GenerationResult
from src.evaluation.metrics import compute_metrics
from src.qrels.filters import CandidateFilter


def build_rag_prompt(query: str, chunks, top_n: int = 5) -> str:
    """RAG 프롬프트 구성: 검색된 청크를 Context로 포맷팅"""
    parts = []
    for i, (chunk, score) in enumerate(chunks[:top_n], 1):
        parts.append(f"[문서 {i}] (Page {chunk.page_number}, Score: {score:.4f})\n{chunk.text}")
    context = "\n\n---\n\n".join(parts)
    return (
        f"다음 문서들을 참고하여 질문에 한국어로 답변해주세요.\n\n"
        f"=== 참고 문서 ===\n{context}\n\n"
        f"=== 질문 ===\n{query}\n\n"
        f"=== 답변 ==="
    )


class ExperimentRunner:
    """
    Gold Qrels 데이터셋을 기준으로 Retriever 성능을 평가하는 실행기.
    질문별 Retrieval → IR 메트릭(Recall, MRR, NDCG) 집계 → (선택) LLM 답변 생성 → DB 저장.
    """

    def __init__(self, run_repo, qrels_repo, question_repo, llm=None, reranker=None):
        self.run_repo      = run_repo
        self.qrels_repo    = qrels_repo
        self.question_repo = question_repo
        self.llm           = llm
        self.reranker      = reranker
        self._filter       = CandidateFilter()  # 텍스트 중복 제거용

    def run(
        self,
        tenant_id: str,
        dataset_id: str,
        config_id: str,
        retriever,
        run_name: str,
        top_k: int = 10,
        use_reranker: bool = False,
        fusion_weight: float = 1.0,
        generate_answers: bool = False,
        progress_callback=None,
    ) -> Tuple[str, Dict[str, float], List[Dict]]:
        """
        실험 실행.
        Returns:
            run_id: str
            avg_metrics: Dict[metric_name -> avg_value]
            per_question: List[Dict] (질문별 상세 결과 + prompt_context + llm_answer)
        """
        run_id = str(uuid.uuid4())
        self.run_repo.save_run(ExperimentRun(
            run_id=run_id, config_id=config_id, dataset_id=dataset_id,
            run_name=run_name, status="RUNNING",
            started_at=datetime.datetime.now()
        ))

        # ── 데이터 로드 ──────────────────────────────────────
        qrels = self.qrels_repo.get_qrels_by_dataset(dataset_id)
        qrels_by_q: Dict[str, Dict[str, int]] = {}
        for q in qrels:
            qrels_by_q.setdefault(q.question_id, {})[q.chunk_id] = q.operator_score

        all_questions = self.question_repo.list_questions(tenant_id)
        question_map  = {q.question_id: q for q in all_questions}

        # ── 질문별 실행 ──────────────────────────────────────
        metric_accum: Dict[str, List[float]] = {}
        per_question: List[Dict] = []
        retrieval_batch: List[RetrievalResult] = []
        generation_batch: List[GenerationResult] = []

        total = len(qrels_by_q)
        for idx, (question_id, qrel_map) in enumerate(qrels_by_q.items()):
            question = question_map.get(question_id)
            if not question:
                continue

            if progress_callback:
                progress_callback(idx, total, question.query_text)

            # 넓은 후보풀 검색 → 텍스트 중복 제거 → (선택) Reranker → top_k
            fetch_k     = top_k * 3 if (use_reranker and self.reranker) else top_k * 2
            raw_results = retriever.retrieve_chunks(tenant_id, question.query_text, top_k=fetch_k)
            deduped     = self._filter.filter_candidates(raw_results)  # 텍스트 중복 제거

            if use_reranker and self.reranker:
                results = self.reranker.rerank(
                    question.query_text, deduped, top_k=top_k, fusion_weight=fusion_weight
                )
            else:
                results = deduped[:top_k]

            retrieved_ids = [chunk.chunk_id for chunk, _ in results]

            for rank, (chunk, score) in enumerate(results, 1):
                retrieval_batch.append(RetrievalResult(
                    run_id=run_id, question_id=question_id,
                    chunk_id=chunk.chunk_id, rank=rank, score=float(score)
                ))

            # ── LLM 답변 생성 (선택) ─────────────────────────
            prompt_context = build_rag_prompt(question.query_text, results, top_n=top_k)
            llm_answer     = ""
            latency_ms     = 0.0
            if generate_answers and self.llm:
                t_start    = datetime.datetime.now()
                llm_answer = self.llm.generate_answer(prompt_context, temperature=0.1)
                latency_ms = (datetime.datetime.now() - t_start).total_seconds() * 1000
                generation_batch.append(GenerationResult(
                    run_id=run_id, question_id=question_id,
                    generated_answer=llm_answer, latency_ms=latency_ms,
                    token_usage=0
                ))

            # 메트릭 계산
            metrics = compute_metrics(retrieved_ids, qrel_map, top_k)
            for name, val in metrics.items():
                metric_accum.setdefault(name, []).append(val)

            relevant_count = sum(1 for s in qrel_map.values() if s >= 1)
            per_question.append({
                "question":        question.query_text,
                "question_id":     question_id,
                "relevant_total":  relevant_count,
                "retrieved_top_k": len(results),
                "retrieved_chunks": results,       # (Chunk, score) 리스트
                "prompt_context":  prompt_context,
                "llm_answer":      llm_answer,
                "latency_ms":      latency_ms,
                **metrics,
            })

        # ── DB 저장 ───────────────────────────────────────────
        self.run_repo.save_retrieval_results(retrieval_batch)
        if generation_batch:
            self.run_repo.save_generation_results(generation_batch)

        avg_metrics = {
            name: sum(vals) / len(vals)
            for name, vals in metric_accum.items() if vals
        }
        self.run_repo.save_evaluation_results([
            EvaluationResult(run_id=run_id, metric_name=name, metric_value=val)
            for name, val in avg_metrics.items()
        ])
        self.run_repo.update_run_status(run_id, "COMPLETED", datetime.datetime.now())

        return run_id, avg_metrics, per_question
