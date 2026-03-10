import uuid
import datetime
import streamlit as st
import pandas as pd

from src.registry import Registry
from src.core.models import ExperimentConfig


def render_runner_tab():
    st.header("🚀 Experiment Runner")
    st.markdown("Gold Qrels 데이터셋을 기준으로 Retriever 성능을 평가합니다.")

    registry = Registry.get_instance()

    # ── 1. Tenant 선택 (사이드바) ──────────────────────────────
    st.sidebar.subheader("🏢 Tenant")
    tenants = registry.doc_repo.list_tenants()
    if not tenants:
        st.warning("등록된 Tenant가 없습니다. 먼저 Document Upload 탭에서 문서를 업로드해주세요.")
        return

    tenant_options = {t.tenant_name: t.tenant_id for t in tenants}
    selected_tenant_name = st.sidebar.selectbox("Tenant 선택", list(tenant_options.keys()), key="runner_tenant")
    tenant_id = tenant_options[selected_tenant_name]

    # ── 2. 실험 선택 (사이드바) ────────────────────────────────
    st.sidebar.divider()
    st.sidebar.subheader("🧪 실험")
    experiments = registry.experiment_repo.list_experiments(tenant_id)
    if not experiments:
        st.warning("등록된 실험이 없습니다. Gold Qrels Builder 탭에서 실험을 생성해주세요.")
        return

    exp_options = {
        (e.get("name") or "<이름 없음>"): str(e["experiment_id"])
        for e in experiments
    }
    selected_exp_name = st.sidebar.selectbox("실험 선택", list(exp_options.keys()), key="runner_exp")
    experiment_id = exp_options[selected_exp_name]
    exp_info  = next((e for e in experiments if str(e["experiment_id"]) == experiment_id), {})
    exp_model = exp_info.get("embedding_model", "bge-m3")
    st.sidebar.caption(f"Embedding Model: `{exp_model}`")

    # ── 3. Dataset 선택 ────────────────────────────────────────
    st.divider()
    st.subheader("📂 Gold Qrels 데이터셋 선택")
    datasets = registry.qrels_repo.list_datasets_by_experiment(experiment_id)

    if not datasets:
        st.warning("이 실험에 저장된 Gold Qrels 데이터셋이 없습니다. Gold Qrels Builder에서 먼저 저장해주세요.")
        return

    dataset_rows = []
    dataset_map  = {}
    for ds in datasets:
        qrels      = registry.qrels_repo.get_qrels_by_dataset(ds.dataset_id)
        q_ids      = {q.question_id for q in qrels}
        dataset_rows.append({
            "데이터셋 이름":  ds.name,
            "버전":           ds.version,
            "질문 수":        len(q_ids),
            "Qrels 수":       len(qrels),
            "생성일":         str(ds.created_at)[:19],
            "Dataset ID":     ds.dataset_id,
        })
        dataset_map[f"{ds.name} ({ds.version})"] = ds.dataset_id

    st.dataframe(pd.DataFrame(dataset_rows), use_container_width=True, hide_index=True)
    selected_ds_label = st.selectbox("평가할 데이터셋 선택", list(dataset_map.keys()))
    dataset_id = dataset_map[selected_ds_label]

    # ── 4. Run 설정 ────────────────────────────────────────────
    st.divider()
    st.subheader("⚙️ Run 설정")

    col1, col2 = st.columns(2)
    with col1:
        run_name = st.text_input("Run 이름", value=f"Run_{datetime.date.today()}")
        # 중복 이름 경고 (soft)
        try:
            existing_runs = registry.run_repo.list_runs(experiment_id)
            dup_names = [r["run_name"] for r in existing_runs]
        except Exception:
            dup_names = []
        if run_name and run_name in dup_names:
            st.warning(f"⚠️ '{run_name}' 이름의 Run이 이미 존재합니다. 저장은 가능하지만 이름을 구분하기 어려울 수 있습니다.")

        retriever_type = st.selectbox(
            "Retriever 타입",
            options=["hybrid", "vector", "bm25"],
            format_func=lambda x: {
                "hybrid": "Hybrid (Vector + BM25, RRF)",
                "vector": "Vector Only (pgvector)",
                "bm25":   "BM25 Only",
            }[x]
        )

    with col2:
        top_k = st.slider("Top-K", min_value=1, max_value=50, value=10)
        use_reranker = st.checkbox("Cross-Encoder Reranker 적용", value=False)
        if use_reranker:
            reranker_model = st.selectbox(
                "Reranker 모델",
                options=[
                    "BAAI/bge-reranker-v2-m3",
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                ],
                format_func=lambda x: {
                    "BAAI/bge-reranker-v2-m3":              "BGE Reranker v2-m3 ✅ 다국어(한국어 지원)",
                    "cross-encoder/ms-marco-MiniLM-L-6-v2": "MS-MARCO MiniLM ⚠️ 영어 전용",
                }[x]
            )
            if reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2":
                st.warning("⚠️ 이 모델은 영어 전용입니다. 한국어 문서에 적용 시 메트릭이 0에 가까워질 수 있습니다.")
        else:
            reranker_model = "BAAI/bge-reranker-v2-m3"
        generate_answers = st.checkbox("🤖 LLM 답변 생성 포함 (Ollama)", value=False)

    # Hybrid 선택 시 가중치 슬라이더
    vector_weight = 0.5
    bm25_weight   = 0.5
    if retriever_type == "hybrid":
        st.markdown("**🔧 Hybrid 가중치 설정** (Vector + BM25 합산 = 1.0)")
        vector_weight = st.slider("Vector 가중치", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        bm25_weight   = round(1.0 - vector_weight, 2)
        st.caption(f"Vector: **{vector_weight}**  |  BM25 (Keyword): **{bm25_weight}**")

    # ── 5. 실행 버튼 ───────────────────────────────────────────
    st.divider()
    if st.button("▶️ 실험 실행", type="primary", use_container_width=True):
        # Config 저장
        config = ExperimentConfig(
            config_id=str(uuid.uuid4()),
            experiment_id=experiment_id,
            retriever_type=retriever_type,
            embedding_model=exp_model,
            reranker_type=reranker_model if use_reranker else None,
            llm_model="llama3.1:8b",
        )
        registry.experiment_repo.save_config(config)

        retriever = registry.create_retriever(retriever_type, exp_model, vector_weight, bm25_weight)
        runner    = registry.create_experiment_runner(reranker_model=reranker_model if use_reranker else None)

        progress_bar  = st.progress(0)
        status_text   = st.empty()

        def progress_callback(idx, total, query_text):
            pct = int((idx / total) * 100)
            progress_bar.progress(pct)
            status_text.caption(f"[{idx+1}/{total}] {query_text[:60]}...")

        with st.spinner("실험 실행 중..."):
            run_id, avg_metrics, per_question = runner.run(
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                config_id=config.config_id,
                retriever=retriever,
                run_name=run_name,
                top_k=top_k,
                use_reranker=use_reranker,
                generate_answers=generate_answers,
                progress_callback=progress_callback,
            )

        progress_bar.progress(100)
        status_text.empty()
        st.success(f"실험 완료! Run ID: `{run_id}`")

        # 결과와 함께 실행 당시의 설정값도 session_state에 저장
        st.session_state["last_run_id"]         = run_id
        st.session_state["last_avg_metrics"]    = avg_metrics
        st.session_state["last_per_question"]   = per_question
        st.session_state["last_top_k"]          = top_k
        st.session_state["last_run_name"]       = run_name
        st.session_state["last_retriever_type"] = retriever_type
        st.session_state["last_vector_weight"]  = vector_weight
        st.session_state["last_bm25_weight"]    = bm25_weight
        st.session_state["last_use_reranker"]   = use_reranker
        st.session_state["last_reranker_model"] = reranker_model if use_reranker else "미적용"
        st.session_state["last_exp_name"]       = selected_exp_name
        st.session_state["last_ds_label"]       = selected_ds_label
        st.session_state["last_exp_model"]      = exp_model

    # ── 6. 결과 표시 ───────────────────────────────────────────
    if "last_avg_metrics" in st.session_state:
        st.divider()
        st.subheader("📊 평가 결과")

        avg_metrics  = st.session_state["last_avg_metrics"]
        per_question = st.session_state["last_per_question"]
        top_k_used   = st.session_state["last_top_k"]
        # 실행 당시 설정값 복원 (현재 위젯 값 대신 session_state에서 읽음)
        _run_name       = st.session_state.get("last_run_name", run_name)
        _retriever_type = st.session_state.get("last_retriever_type", retriever_type)
        _vector_weight  = st.session_state.get("last_vector_weight", vector_weight)
        _bm25_weight    = st.session_state.get("last_bm25_weight", bm25_weight)
        _reranker_model = st.session_state.get("last_reranker_model", "미적용")
        _exp_name       = st.session_state.get("last_exp_name", selected_exp_name)
        _ds_label       = st.session_state.get("last_ds_label", selected_ds_label)
        _exp_model      = st.session_state.get("last_exp_model", exp_model)

        # ── 메트릭 카드 ─────────────────────────────────────
        cols = st.columns(len(avg_metrics))
        for col, (name, val) in zip(cols, avg_metrics.items()):
            col.metric(label=name, value=f"{val:.4f}")

        # ── 결과 요약 (복사용) ───────────────────────────────
        import datetime as _dt
        reranker_label = _reranker_model
        weight_label   = (f"Vector {_vector_weight} / BM25 {_bm25_weight}"
                          if _retriever_type == "hybrid" else "-")
        pq_lines = "\n".join(
            f"  Q{i+1}. {pq['question'][:50]}{'...' if len(pq['question'])>50 else ''}\n"
            f"       관련문서: {pq['relevant_total']}개  |  " +
            "  ".join(f"{k}: {v:.4f}" for k, v in pq.items()
                      if k.startswith(("Recall","MRR","NDCG")))
            for i, pq in enumerate(per_question)
        )
        summary_text = f"""====== RAG Retrieval 평가 결과 요약 ======
Run       : {_run_name}
일시       : {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}
실험       : {_exp_name}
데이터셋   : {_ds_label}

[검색 설정]
Retriever  : {_retriever_type.upper()}
가중치     : {weight_label}
Top-K      : {top_k_used}
Reranker   : {reranker_label}
Embedding  : {_exp_model}

[평균 메트릭]
""" + "\n".join(f"  {k:<12}: {v:.4f}" for k, v in avg_metrics.items()) + f"""

[질문별 상세]
{pq_lines}
=========================================="""

        with st.expander("📋 결과 요약 — 우측 아이콘 클릭으로 복사", expanded=True):
            st.code(summary_text, language=None)

        # ── 메트릭 해석 ─────────────────────────────────────
        with st.expander("📈 메트릭 해석 및 개선 가이드", expanded=True):
            for name, val in avg_metrics.items():
                if name.startswith("Recall"):
                    if val >= 0.8:
                        level, advice = "🟢 양호", "관련 문서의 대부분을 검색하고 있습니다."
                    elif val >= 0.5:
                        level, advice = "🟡 보통", "Top-K를 늘리거나 Hybrid 가중치(Vector↑)를 조정해보세요."
                    else:
                        level, advice = "🔴 낮음", "검색된 관련 문서 비율이 낮습니다. 청킹 전략 재검토 또는 Embedding 모델 교체를 권장합니다."
                    st.markdown(f"**{name} = {val:.4f}** — {level}  \n→ {advice}")

                elif name.startswith("MRR"):
                    first_rank = 1 / val if val > 0 else float("inf")
                    if val >= 0.5:
                        level, advice = "🟢 양호", f"첫 번째 관련 문서가 평균 {first_rank:.1f}위에 위치합니다."
                    elif val >= 0.25:
                        level, advice = "🟡 보통", f"첫 번째 관련 문서가 평균 {first_rank:.1f}위입니다. Cross-Encoder Reranker 적용을 권장합니다."
                    else:
                        level, advice = "🔴 낮음", f"첫 번째 관련 문서가 평균 {first_rank:.1f}위로 매우 낮습니다. Retriever 타입 변경(Vector↔BM25)을 시도해보세요."
                    st.markdown(f"**{name} = {val:.4f}** — {level}  \n→ {advice}")

                elif name.startswith("NDCG"):
                    if val >= 0.7:
                        level, advice = "🟢 양호", "상위 랭크에 관련 문서가 잘 배치되어 있습니다."
                    elif val >= 0.4:
                        level, advice = "🟡 보통", "관련 문서의 랭킹 품질이 보통입니다. Reranker 또는 가중치 조정을 시도하세요."
                    else:
                        level, advice = "🔴 낮음", "관련 문서가 하위 랭크에 집중되어 있습니다. Embedding 모델 교체 또는 Reranker 적용을 강력히 권장합니다."
                    st.markdown(f"**{name} = {val:.4f}** — {level}  \n→ {advice}")

        # ── 질문별 상세 테이블 ───────────────────────────────
        st.markdown("**질문별 상세 결과**")
        detail_rows = []
        for pq in per_question:
            row = {
                "질문":        pq["question"][:60] + ("..." if len(pq["question"]) > 60 else ""),
                "관련 문서 수": pq["relevant_total"],
                "검색 수":     pq["retrieved_top_k"],
            }
            row.update({k: f"{v:.4f}" for k, v in pq.items()
                        if k.startswith(("Recall", "MRR", "NDCG"))})
            detail_rows.append(row)
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

        # ── 질문별 Prompt Context + LLM 답변 ────────────────
        st.divider()
        st.markdown("**🔍 질문별 Prompt Context & 검색 결과 상세**")
        for i, pq in enumerate(per_question):
            label = f"Q{i+1}. {pq['question'][:70]}{'...' if len(pq['question']) > 70 else ''}"
            with st.expander(label, expanded=(i == 0)):
                # Prompt Context
                st.markdown("**📋 Prompt Context (LLM 입력 전문)**")
                st.code(pq["prompt_context"], language="markdown")

                # LLM 답변
                if pq.get("llm_answer"):
                    st.markdown("**🤖 LLM 답변**")
                    st.markdown(
                        f'<div style="background:#f0f4ff;color:#1a1a2e;border-left:4px solid #4a6cf7;'
                        f'padding:12px 16px;border-radius:4px;white-space:pre-wrap;font-size:0.95em;">'
                        f'{pq["llm_answer"]}</div>',
                        unsafe_allow_html=True
                    )
                    if pq.get("latency_ms"):
                        st.caption(f"생성 시간: {pq['latency_ms']:.0f}ms")
                else:
                    st.info("LLM 답변 생성이 비활성화되어 있습니다. Run 설정에서 '🤖 LLM 답변 생성 포함'을 체크하세요.")

                # 검색된 청크 목록
                st.markdown("**📄 검색된 청크 (상위 순)**")
                for rank, (chunk, score) in enumerate(pq.get("retrieved_chunks", []), 1):
                    is_relevant = pq.get  # 아래에서 qrel 정보 없어 표시 생략
                    st.markdown(
                        f"`Rank {rank}` | Score: `{score:.4f}` | Page {chunk.page_number} | "
                        f"Chunk ID: `{chunk.chunk_id[:8]}...`"
                    )
                    st.text(chunk.text[:300] + ("..." if len(chunk.text) > 300 else ""))

    # ── 7. 이전 Run 목록 ───────────────────────────────────────
    st.divider()
    st.subheader("📋 이전 실행 목록")
    try:
        runs = registry.run_repo.list_runs(experiment_id)
    except Exception:
        runs = []

    if not runs:
        st.info("아직 실행된 Run이 없습니다.")
    else:
        run_rows = []
        for r in runs:
            metrics = registry.run_repo.get_run_metrics(str(r["run_id"]))
            metric_str = "  |  ".join(f"{m['metric_name']}: {m['metric_value']:.4f}" for m in metrics)
            run_rows.append({
                "Run 이름":       r["run_name"],
                "Retriever":      r.get("retriever_type", "-"),
                "Embedding":      r.get("embedding_model", "-"),
                "상태":           r["status"],
                "시작":           str(r["started_at"])[:19],
                "종료":           str(r["finished_at"])[:19] if r["finished_at"] else "-",
                "메트릭":         metric_str,
            })
        st.dataframe(pd.DataFrame(run_rows), use_container_width=True, hide_index=True)
