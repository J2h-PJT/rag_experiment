import streamlit as st
import pandas as pd
import datetime
import time
import json
import requests
import difflib
from typing import List, Dict, Tuple, Optional

from src.registry import Registry
from src.core.models import Chunk
from src.evaluation.metrics import recall_at_k, mrr_at_k, ndcg_at_k


def render_chat_tab():
    """RAG Chat 탭 — Gold Qrels 검증 기반 대화 인터페이스"""
    st.header("💬 RAG Chat — Gold Qrels 검증")
    st.markdown("실험한 Retriever 설정값으로 대화하며 검증합니다.")

    registry = Registry.get_instance()

    # ── 초기화 ───────────────────────────────────────────────────────────
    _init_session_state()

    # ── 1. Tenant 선택 (사이드바) ──────────────────────────────────────
    st.sidebar.subheader("🏢 Tenant")
    tenants = registry.doc_repo.list_tenants()
    if not tenants:
        st.warning("등록된 Tenant가 없습니다.")
        return

    tenant_options = {t.tenant_name: t.tenant_id for t in tenants}
    selected_tenant_name = st.sidebar.selectbox(
        "Tenant 선택", list(tenant_options.keys()), key="chat_tenant"
    )
    tenant_id = tenant_options[selected_tenant_name]
    st.session_state.chat["tenant_id"] = tenant_id

    # ── 2. Experiment 선택 (사이드바) ────────────────────────────────
    st.sidebar.divider()
    st.sidebar.subheader("🧪 실험")
    experiments = registry.experiment_repo.list_experiments(tenant_id)
    if not experiments:
        st.warning("등록된 실험이 없습니다.")
        return

    exp_options = {(e.get("name") or "<이름 없음>"): str(e["experiment_id"]) for e in experiments}
    selected_exp_name = st.sidebar.selectbox("실험 선택", list(exp_options.keys()), key="chat_exp")
    experiment_id = exp_options[selected_exp_name]
    exp_info = next((e for e in experiments if str(e["experiment_id"]) == experiment_id), {})
    exp_model = exp_info.get("embedding_model", "bge-m3")
    st.session_state.chat["experiment_id"] = experiment_id
    st.sidebar.caption(f"Embedding Model: `{exp_model}`")

    # ── 3. Dataset 선택 ──────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.subheader("📂 Dataset")
    datasets = registry.qrels_repo.list_datasets_by_experiment(experiment_id)
    if not datasets:
        st.warning("이 실험에 Gold Qrels 데이터셋이 없습니다.")
        return

    dataset_options = {f"{ds.name} ({ds.version})": ds.dataset_id for ds in datasets}
    selected_ds_label = st.sidebar.selectbox("Dataset 선택", list(dataset_options.keys()), key="chat_dataset")
    dataset_id = dataset_options[selected_ds_label]
    st.session_state.chat["dataset_id"] = dataset_id

    # ── 4. Run 선택 ─────────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.subheader("🚀 Run (설정값)")
    try:
        runs = registry.run_repo.list_runs(experiment_id)
        if not runs:
            st.sidebar.warning("이 실험의 Run이 없습니다. Experiment Runner 탭에서 실험을 실행하세요.")
            return
    except Exception as e:
        st.sidebar.error(f"Run 로드 실패: {e}")
        return

    run_labels = [f"{r['run_name']} ({r.get('retriever_type', 'hybrid').upper()})" for r in runs]
    selected_run_idx = st.sidebar.selectbox("Run 선택", range(len(runs)), format_func=lambda i: run_labels[i], key="chat_run")
    selected_run = runs[selected_run_idx]
    run_id = str(selected_run["run_id"])

    # ── Run 설정값 로드 ───────────────────────────────────────────────
    run_config = _load_run_config(selected_run, exp_model)

    # ── 5. 검색 파라미터 오버라이드 (사이드바) ─────────────────────────
    # top_k, vector_weight, keyword_weight는 DB에 저장되지 않으므로 사이드바에서 설정
    st.sidebar.divider()
    st.sidebar.subheader("⚙️ 검색 파라미터")
    top_k = st.sidebar.slider("Top-K", min_value=1, max_value=50, value=run_config.get("top_k", 10), key="chat_top_k")
    run_config["top_k"] = top_k

    if run_config.get("retriever_type") == "hybrid":
        vector_weight = st.sidebar.slider(
            "Vector 가중치", min_value=0.0, max_value=1.0,
            value=run_config.get("vector_weight", 0.5), step=0.05, key="chat_vw"
        )
        bm25_weight = round(1.0 - vector_weight, 2)
        st.sidebar.caption(f"BM25 가중치: **{bm25_weight}**")
        run_config["vector_weight"] = vector_weight
        run_config["keyword_weight"] = bm25_weight

    if run_config.get("reranker_model"):
        st.sidebar.divider()
        st.sidebar.subheader("🔀 Reranker Fusion")
        saved_fw = run_config.get("fusion_weight", 1.0)
        fusion_weight = st.sidebar.slider(
            "Reranker 가중치",
            min_value=0.0, max_value=1.0,
            value=float(saved_fw), step=0.1,
            key="chat_fusion",
            help="1.0=순수 Reranker / 0.5=Retrieval+Reranker 혼합 / 0.0=Reranker 미적용"
        )
        st.sidebar.caption(
            {1.0: "🔴 순수 Reranker", 0.0: "🟢 Retrieval 점수 유지"}.get(
                fusion_weight,
                f"🟡 Retrieval {(1-fusion_weight):.0%} + Reranker {fusion_weight:.0%}"
            )
        )
        run_config["fusion_weight"] = fusion_weight
    else:
        run_config["fusion_weight"] = 0.0

    st.session_state.chat["run_config"] = run_config

    # ── Context 변경 감지 → Chat history 초기화 ───────────────────────
    _prev_ctx = st.session_state.chat.get("_chat_ctx")
    _cur_ctx = f"{tenant_id}|{experiment_id}|{dataset_id}|{run_id}"
    if _prev_ctx is not None and _prev_ctx != _cur_ctx:
        st.session_state.chat["messages"] = []
        st.session_state.chat["session_summary"] = {
            "query_count": 0,
            "validated_count": 0,
            "total_precision": 0.0,
            "total_recall": 0.0,
            "total_mrr": 0.0,
            "total_ndcg": 0.0,
        }
        st.info("🔄 Context 변경되어 Chat history가 초기화되었습니다.")
    st.session_state.chat["_chat_ctx"] = _cur_ctx

    # ── Gold Qrels 로드 ───────────────────────────────────────────────
    try:
        qrels = registry.qrels_repo.get_qrels_by_dataset(dataset_id)
        st.session_state.chat["gold_qrels"] = qrels
    except Exception as e:
        st.error(f"Gold Qrels 로드 실패: {e}")
        return

    # ── 질문 텍스트→question_id 매핑 로드 ────────────────────────────
    # qrels_tab은 uuid4()로 question_id를 저장하므로 query_text로 매칭해야 함
    try:
        all_questions = registry.question_repo.list_questions(tenant_id)
        # 현재 experiment에 속한 질문만 필터링
        question_map = {
            q.query_text: q.question_id
            for q in all_questions
            if q.experiment_id == experiment_id
        }
    except Exception as e:
        st.warning(f"질문 매핑 로드 실패 (검증 비활성화): {e}")
        question_map = {}

    # ── 메인 영역 ──────────────────────────────────────────────────────
    col_main, col_sidebar = st.columns([0.75, 0.25])

    with col_main:
        # ── Config Panel ──────────────────────────────────────────────
        _render_config_panel(run_config, dataset_id, qrels, question_map)

        # ── Chat History ──────────────────────────────────────────────
        st.divider()
        _render_chat_history(st.session_state.chat["messages"])

        # ── Chat Input ────────────────────────────────────────────────
        st.divider()
        _render_chat_input_and_send(registry, run_config, qrels, tenant_id, exp_model, question_map)

    with col_sidebar:
        # ── Session Summary ───────────────────────────────────────────
        _render_session_summary(st.session_state.chat["messages"])


def _init_session_state():
    """Chat session 초기화"""
    if "chat" not in st.session_state:
        st.session_state.chat = {
            "tenant_id": None,
            "experiment_id": None,
            "dataset_id": None,
            "run_id": None,
            "run_config": {},
            "gold_qrels": [],
            "messages": [],
            "session_summary": {
                "query_count": 0,
                "validated_count": 0,
                "total_precision": 0.0,
                "total_recall": 0.0,
                "total_mrr": 0.0,
                "total_ndcg": 0.0,
            },
            "_chat_ctx": None,
        }


def _load_run_config(run_data: dict, exp_model: str) -> dict:
    """Run 데이터로부터 설정값 로드.
    reranker_type=NULL이면 Run이 Reranker 미적용으로 실행된 것 → None 유지.
    """
    reranker = run_data.get("reranker_type")  # NULL이면 None 그대로 유지
    llm = run_data.get("llm_model") or "llama3.1:8b"
    return {
        "run_id": str(run_data.get("run_id", "")),
        "run_name": run_data.get("run_name", "Unknown"),
        "retriever_type": run_data.get("retriever_type", "hybrid"),
        "embedding_model": exp_model,
        "top_k": 10,  # DB에 없으므로 기본값 (사이드바 슬라이더로 오버라이드)
        "vector_weight": 0.5,
        "keyword_weight": 0.5,
        "reranker_model": reranker,  # None = Reranker 미적용
        "llm_model": llm,
        "fusion_weight": run_data.get("fusion_weight") if run_data.get("fusion_weight") is not None else 1.0,
    }


def _render_config_panel(config: dict, dataset_id: str, qrels: list, question_map: dict):
    """상단 설정 패널 렌더링"""
    st.markdown("### ✨ Run Configuration")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Run Name", config.get("run_name", "N/A")[:30])

    with col2:
        retriever_display = {
            "hybrid": "Hybrid",
            "vector": "Vector",
            "bm25": "BM25",
        }.get(config.get("retriever_type", "hybrid"), "Unknown")
        st.metric("Retriever", retriever_display)

    with col3:
        st.metric("Top-K", config.get("top_k", 10))

    with col4:
        reranker_val = config.get("reranker_model")
        if not reranker_val:
            st.metric("Reranker", "미적용")
        else:
            fw = config.get("fusion_weight", 1.0)
            label = f"{reranker_val.split('/')[-1][:15]} (α={fw:.1f})"
            st.metric("Reranker", label)

    # Hybrid 가중치
    if config.get("retriever_type") == "hybrid":
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Vector Weight", f"{config.get('vector_weight', 0.5):.2f}")
        with col2:
            st.metric("Keyword Weight", f"{config.get('keyword_weight', 0.5):.2f}")

    # Dataset 정보
    st.divider()
    q_ids = {q.question_id for q in qrels}
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Questions in Dataset", len(q_ids))
    with col2:
        st.metric("Total Qrels", len(qrels))
    with col3:
        # question_map에 있는 질문 중 실제 qrels가 있는 것
        matchable = sum(1 for qt in question_map if any(q.question_id == question_map[qt] for q in qrels))
        st.metric("검증 가능 질문", matchable)

    if question_map:
        with st.expander("📋 Gold Qrels 질문 목록 (채팅에서 그대로 입력하면 검증됩니다)"):
            for qt in question_map:
                q_id = question_map[qt]
                cnt = sum(1 for q in qrels if q.question_id == q_id and q.operator_score >= 1)
                st.markdown(f"- **{qt}** → 관련 청크 {cnt}개")


def _render_chat_history(messages: list):
    """Chat history 렌더링"""
    if not messages:
        st.info("💬 아직 메시지가 없습니다. 질문을 입력해주세요.")
        return

    chat_container = st.container(border=True)
    with chat_container:
        for msg in messages:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
                    st.caption(f"🕐 {msg['timestamp'].strftime('%H:%M:%S')}")

            else:  # assistant
                with st.chat_message("assistant"):
                    st.write(msg["content"])

                    # Retrieved Chunks
                    if msg.get("retrieved_chunks"):
                        with st.expander("📎 검색 결과 & 검증"):
                            for i, chunk in enumerate(msg["retrieved_chunks"], 1):
                                col1, col2, col3 = st.columns([0.6, 0.2, 0.2])

                                validation_info = None
                                if msg.get("validation"):
                                    matched = next(
                                        (m for m in msg["validation"].get("matched_chunks", [])
                                         if m["chunk_id"] == chunk.get("chunk_id")),
                                        None
                                    )
                                    validation_info = matched

                                with col1:
                                    status_icon = "✅" if (validation_info and validation_info.get("is_relevant")) else "❌"
                                    st.write(f"**{i}.** {status_icon} (p.{chunk.get('page', 'N/A')})")
                                    preview = chunk.get("text", "")[:200]
                                    if len(chunk.get("text", "")) > 200:
                                        preview += "..."
                                    st.code(preview, language="text")

                                with col2:
                                    score = chunk.get("score", 0)
                                    st.metric("Score", f"{score:.4f}")

                                with col3:
                                    if validation_info:
                                        qrel_status = "YES" if validation_info.get("is_relevant") else "NO"
                                        st.metric("Gold Qrels", qrel_status)

                    # Validation Metrics
                    if msg.get("validation"):
                        st.divider()
                        validation = msg["validation"]
                        if validation.get("has_qrels"):
                            sim = validation.get("similarity", 1.0)
                            matched_q = validation.get("matched_question", "")
                            if sim < 1.0 and matched_q:
                                st.caption(f"🔗 매칭된 Gold Qrels 질문: **\"{matched_q}\"** (유사도 {sim:.0%})")
                            st.write("**📊 이 쿼리의 성능 지표** (Gold Qrels 기준)")
                            metrics = validation.get("metrics", {})
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Precision", f"{metrics.get('precision', 0):.2%}")
                            with col2:
                                st.metric("Recall", f"{metrics.get('recall', 0):.2%}")
                            with col3:
                                st.metric("MRR", f"{metrics.get('mrr', 0):.4f}")
                            with col4:
                                st.metric("NDCG", f"{metrics.get('ndcg', 0):.4f}")
                        else:
                            nearest = validation.get("matched_question")
                            sim = validation.get("similarity", 0.0)
                            if nearest:
                                st.caption(f"⚪ Gold Qrels 매칭 없음 (가장 유사: \"{nearest}\", 유사도 {sim:.0%}) — RAG 답변만 제공")
                            else:
                                st.caption("⚪ Gold Qrels에 등록된 질문 없음 — RAG 답변만 제공")

                    # Footer
                    processing_time = msg.get("processing_time", 0)
                    st.caption(
                        f"🕐 {msg['timestamp'].strftime('%H:%M:%S')} | "
                        f"⏱️ {processing_time:.1f}s"
                    )


def _render_chat_input_and_send(
    registry, run_config: dict, qrels: list, tenant_id: str, exp_model: str, question_map: dict
):
    """Chat 입력창 및 전송 로직"""
    col1, col2 = st.columns([0.9, 0.1])

    with col1:
        user_input = st.text_input(
            "💬 메시지 입력",
            placeholder="질문을 입력하세요... (Gold Qrels에 등록된 질문을 입력하면 검증 메트릭이 표시됩니다)",
            key="chat_input",
        )

    with col2:
        send_button = st.button("📤", key="chat_send_btn", use_container_width=True)

    if send_button and user_input:
        # User message 저장
        st.session_state.chat["messages"].append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.datetime.now(),
        })

        # Message processing
        _process_user_message_with_validation(
            registry, user_input, run_config, qrels, tenant_id, exp_model, question_map
        )

        st.rerun()


def _process_user_message_with_validation(
    registry, query: str, run_config: dict, qrels: list, tenant_id: str, exp_model: str, question_map: dict
):
    """User 메시지 처리 + Gold Qrels 검증"""
    start_time = time.time()

    # ① Retrieval
    with st.status("🔍 검색 및 답변 생성 중...", expanded=True) as status:
        s1 = st.empty()
        s1.write("🔍 **[1/3] Hybrid Retrieval** 실행 중...")
        try:
            retriever = registry.create_retriever(
                run_config["retriever_type"],
                exp_model,
                vector_weight=run_config.get("vector_weight", 0.5),
                bm25_weight=run_config.get("keyword_weight", 0.5),
            )
            fetch_k = run_config.get("top_k", 10) * 2
            retrieved: List[Tuple[Chunk, float]] = retriever.retrieve_chunks(
                tenant_id, query, top_k=fetch_k
            )
            s1.write(f"✅ **[1/3] Retrieval** 완료 — {len(retrieved)}개 후보")
        except Exception as e:
            st.error(f"🔍 검색 실패: {e}")
            status.update(label="❌ 검색 실패", state="error")
            return

        # ② Reranking (reranker_model=None이면 스킵)
        s2 = st.empty()
        reranker_model = run_config.get("reranker_model")
        top_k = run_config.get("top_k", 10)
        if reranker_model:
            s2.write(f"🎯 **[2/3] Reranking** 중... ({reranker_model.split('/')[-1]})")
            try:
                reranker = registry.create_reranker(reranker_model)
                fusion_weight = run_config.get("fusion_weight", 1.0)
                ranked: List[Tuple[Chunk, float]] = reranker.rerank(
                    query, retrieved, top_k=top_k, fusion_weight=fusion_weight
                )
                s2.write(f"✅ **[2/3] Reranking** 완료 — 상위 {len(ranked)}개 선택")
            except Exception as e:
                st.error(f"🎯 Reranking 실패: {e}")
                status.update(label="❌ Reranking 실패", state="error")
                return
        else:
            ranked = retrieved[:top_k]
            s2.write(f"⏭️ **[2/3] Reranking 미적용** — 상위 {len(ranked)}개 선택 (Run 설정 기준)")

        # ③ Validation against Gold Qrels
        validation_result = _validate_against_qrels(query, ranked, qrels, question_map)

        # ④ Context 구성 및 Prompt 생성
        context_parts = []
        for chunk, score in ranked:
            context_parts.append(f"[p.{chunk.page_number}]\n{chunk.text}")
        context = "\n\n".join(context_parts)

        system_prompt = """당신은 문서 기반 질문-답변 봇입니다.
제공된 문서만을 근거로 답변하세요.
문서에 답변이 없으면 '문서에서 찾을 수 없습니다'라고 명확히 답하세요."""

        full_prompt = f"""{system_prompt}

===== 문서 =====
{context}

===== 질문 =====
{query}

===== 답변 ====="""

        # ⑤ Streaming LLM
        s3 = st.empty()
        s3.write(f"🤖 **[3/3] LLM 답변 생성** 중...")
        response_area = st.empty()
        full_response = ""
        token_count = 0

        try:
            for token in _llm_stream(full_prompt, run_config.get("llm_model", "llama3.1:8b")):
                full_response += token
                token_count += 1
                if token_count % 5 == 0:
                    response_area.markdown(full_response + "▌")
        except Exception as e:
            st.error(f"🤖 LLM 오류: {e}")
            full_response = f"[오류: {e}]"

        response_area.empty()
        s3.write(f"✅ **[3/3] 답변 생성** 완료 ({token_count} 토큰)")
        status.update(label="✅ 완료", state="complete")

    # ⑥ Message 저장
    processing_time = time.time() - start_time

    st.session_state.chat["messages"].append({
        "role": "assistant",
        "content": full_response,
        "timestamp": datetime.datetime.now(),
        "retrieved_chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "score": float(score),
                "page": chunk.page_number,
            }
            for chunk, score in ranked
        ],
        "validation": validation_result,
        "processing_time": processing_time,
    })

    # ⑦ Session summary 업데이트 (Gold Qrels가 있는 쿼리만 메트릭 누적)
    summary = st.session_state.chat["session_summary"]
    summary["query_count"] += 1
    if validation_result.get("has_qrels"):
        metrics = validation_result.get("metrics", {})
        summary["total_precision"] += metrics.get("precision", 0)
        summary["total_recall"] += metrics.get("recall", 0)
        summary["total_mrr"] += metrics.get("mrr", 0)
        summary["total_ndcg"] += metrics.get("ndcg", 0)
        summary["validated_count"] = summary.get("validated_count", 0) + 1


def _find_best_question(query: str, question_map: dict, threshold: float = 0.5):
    """query_text와 가장 유사한 Gold Qrels 질문을 찾아 반환.
    Returns: (question_id, matched_text, similarity_ratio) or (None, None, best_ratio)
    """
    if not question_map:
        return None, None, 0.0

    query_norm = query.strip()

    # 1. 정확 일치
    for qt, qid in question_map.items():
        if qt.strip() == query_norm:
            return qid, qt, 1.0

    # 2. 유사도 기반 매칭 (difflib)
    best_ratio = 0.0
    best_qid = None
    best_qt = None
    for qt, qid in question_map.items():
        ratio = difflib.SequenceMatcher(None, query_norm, qt.strip()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_qid = qid
            best_qt = qt

    if best_ratio >= threshold:
        return best_qid, best_qt, best_ratio

    return None, best_qt, best_ratio


def _validate_against_qrels(query: str, ranked: List[Tuple[Chunk, float]], qrels: list, question_map: dict) -> dict:
    """검색 결과를 Gold Qrels와 비교하여 검증.
    question_map: {query_text: question_id} — qrels_tab이 저장한 uuid4 기반 ID와 매칭
    유사도 기반 매칭(threshold=0.5)으로 자연어 표현 차이를 허용.
    """
    question_id, matched_text, similarity = _find_best_question(query, question_map)

    if not question_id:
        # Gold Qrels에 없는 질문 — 검증 불가
        return {
            "matched_chunks": [],
            "has_qrels": False,
            "matched_question": matched_text,
            "similarity": similarity,
            "metrics": {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0},
        }

    # 이 질문에 해당하는 Gold Qrels (operator_score >= 1인 것만 정답)
    question_qrels = [q for q in qrels if q.question_id == question_id and q.operator_score >= 1]
    qrel_dict = {q.chunk_id: q for q in question_qrels}

    # Matched 정보 구성
    matched_chunks = []
    for rank, (chunk, score) in enumerate(ranked, 1):
        qrel = qrel_dict.get(chunk.chunk_id)
        matched_chunks.append({
            "chunk_id": chunk.chunk_id,
            "is_relevant": qrel is not None,
            "qrels_score": qrel.operator_score if qrel else None,
            "retrieved_rank": rank,
            "retrieved_score": float(score),
        })

    # Metrics 계산
    retrieved_chunk_ids = [chunk.chunk_id for chunk, _ in ranked]
    relevant_chunk_ids = [q.chunk_id for q in question_qrels]
    qrel_scores = {q.chunk_id: q.operator_score for q in qrels if q.question_id == question_id}

    k = len(ranked)

    matched_count = sum(1 for m in matched_chunks if m["is_relevant"])
    precision = matched_count / len(ranked) if ranked else 0.0
    recall = recall_at_k(retrieved_chunk_ids, relevant_chunk_ids, k) if relevant_chunk_ids else 0.0
    mrr = mrr_at_k(retrieved_chunk_ids, relevant_chunk_ids, k)
    ndcg = ndcg_at_k(retrieved_chunk_ids, qrel_scores, k) if qrel_scores else 0.0

    has_qrels = len(question_qrels) > 0

    return {
        "matched_chunks": matched_chunks,
        "has_qrels": has_qrels,
        "matched_question": matched_text,
        "similarity": similarity,
        "metrics": {
            "precision": precision,
            "recall": recall,
            "mrr": mrr,
            "ndcg": ndcg,
        },
    }


def _llm_stream(prompt: str, model: str = "llama3.1:8b"):
    """Ollama HTTP streaming generator"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": {"temperature": 0.3},
            },
            stream=True,
            timeout=300,
        )

        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                except json.JSONDecodeError:
                    pass

    except Exception as e:
        yield f"\n[Error: {e}]"


def _render_session_summary(messages: list):
    """우측 사이드바 — Session 검증 요약"""
    st.subheader("📈 검증 요약")

    summary = st.session_state.chat.get("session_summary", {})
    query_count = summary.get("query_count", 0)
    validated_count = summary.get("validated_count", 0)

    if query_count == 0:
        st.info("아직 쿼리가 없습니다.")
        return

    st.write(f"**총 쿼리**: {query_count}개")
    st.write(f"**검증된 쿼리**: {validated_count}개 (Gold Qrels 매칭)")
    st.write(f"**자유 질문**: {query_count - validated_count}개")
    st.divider()

    if validated_count > 0:
        avg_precision = summary["total_precision"] / validated_count
        avg_recall = summary["total_recall"] / validated_count
        avg_mrr = summary["total_mrr"] / validated_count
        avg_ndcg = summary["total_ndcg"] / validated_count

        st.caption("📊 Gold Qrels 검증 평균 (검증된 쿼리 기준)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Precision", f"{avg_precision:.2%}")
            st.metric("Avg Recall", f"{avg_recall:.2%}")
        with col2:
            st.metric("Avg MRR", f"{avg_mrr:.4f}")
            st.metric("Avg NDCG", f"{avg_ndcg:.4f}")
    else:
        st.info("Gold Qrels에 등록된 질문이 아직 없습니다.\n자유 질문은 RAG 답변만 제공됩니다.")

    st.divider()
    st.subheader("쿼리 이력")

    for i, msg in enumerate(messages):
        if msg["role"] == "assistant" and msg.get("validation"):
            validation = msg["validation"]
            metrics = validation.get("metrics", {})
            has_qrels = validation.get("has_qrels", False)

            user_msg = None
            for j in range(i - 1, -1, -1):
                if messages[j]["role"] == "user":
                    user_msg = messages[j]
                    break

            query_text = (user_msg["content"][:40] + "...") if user_msg else "N/A"
            qrels_badge = "🟢 Qrels 있음" if has_qrels else "⚪ Qrels 없음"
            st.write(f"**Q**: {query_text}")
            st.caption(f"{qrels_badge}")
            if has_qrels:
                st.caption(
                    f"P: {metrics.get('precision', 0):.2%} | "
                    f"R: {metrics.get('recall', 0):.2%} | "
                    f"MRR: {metrics.get('mrr', 0):.4f}"
                )
            st.divider()
