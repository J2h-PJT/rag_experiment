import streamlit as st
import uuid
import datetime
from src.registry import Registry, AVAILABLE_EMBEDDING_MODELS
from src.core.models import Question, DatasetVersion, GoldQrel

def render_qrels_tab():
    st.header("🎯 Gold Qrels 생성기")
    st.markdown("""
        실험용 정답 셋(Ground Truth)을 구축합니다.
        하이브리드 검색(Vector + BM25)과 Cross-Encoder Reranking을 거쳐
        휴먼 인 더 루프(HITL) 방식으로 정답을 확정합니다.
    """)

    registry = Registry.get_instance()

    # ── 1. Tenant 선택 (사이드바) ──────────────────────────
    st.sidebar.subheader("🏢 Tenant")
    tenants = registry.doc_repo.list_tenants()
    if not tenants:
        st.sidebar.warning("등록된 Tenant가 없습니다. 먼저 문서를 업로드해주세요.")
        st.warning("Document Upload 탭에서 Tenant를 먼저 생성해주세요.")
        return

    tenant_options = {t.tenant_name: t.tenant_id for t in tenants}
    selected_tenant_name = st.sidebar.selectbox("Tenant 선택", list(tenant_options.keys()), key="qrels_tenant_sel")
    tenant_id = tenant_options[selected_tenant_name]
    st.sidebar.caption(f"ID: `{tenant_id}`")

    # ── 2. 실험 선택 / 생성 ────────────────────────────────
    st.sidebar.divider()
    st.sidebar.subheader("🧪 실험")

    experiments = registry.experiment_repo.list_experiments(tenant_id)
    exp_options = {
        (e.get("name") or e.get("experiment_name") or "<이름 없음>"): str(e["experiment_id"])
        for e in experiments
    }
    display_options = list(exp_options.keys()) + ["+ 새 실험 생성"]
    selected_exp = st.sidebar.selectbox("실험 선택", display_options, key="qrels_exp_sel")

    experiment_id = None
    if selected_exp == "+ 새 실험 생성":
        with st.sidebar.form("new_exp_form"):
            new_name = st.text_input("실험 이름")
            new_desc = st.text_area("실험 설명")
            model_keys   = list(AVAILABLE_EMBEDDING_MODELS.keys())
            model_labels = [f"{k} — {v}" for k, v in AVAILABLE_EMBEDDING_MODELS.items()]
            new_model_idx = st.selectbox(
                "Embedding Model",
                options=range(len(model_keys)),
                format_func=lambda i: model_labels[i]
            )
            new_model = model_keys[new_model_idx]
            if st.form_submit_button("실험 생성"):
                if not new_name.strip():
                    st.error("실험 이름을 입력해주세요.")
                else:
                    experiment_id = str(uuid.uuid4())
                    registry.experiment_repo.create_experiment(
                        experiment_id, tenant_id, new_name.strip(), new_desc, embedding_model=new_model
                    )
                    # ★ selectbox를 새 실험으로 전환한 뒤 rerun
                    st.session_state["qrels_exp_sel"] = new_name.strip()
                    st.rerun()
        st.warning("실험을 먼저 생성해주세요.")
        return
    else:
        experiment_id = exp_options[selected_exp]
        exp_info      = next((e for e in experiments if str(e["experiment_id"]) == experiment_id), {})
        exp_model     = exp_info.get("embedding_model", "bge-m3")
        st.sidebar.caption(f"Embedding Model: `{exp_model}`")

    # ── Tenant/실험 전환 감지 → stale candidates 클리어 ────
    _prev_ctx = st.session_state.get("qrels_ctx")
    _cur_ctx  = f"{tenant_id}|{experiment_id}"
    if _prev_ctx is not None and _prev_ctx != _cur_ctx:
        for _k in ["qrels_candidates", "current_query", "current_q_id",
                   "current_exp_id", "llm_scores", "qrels_num_candidates", "qrels_exp_model"]:
            st.session_state.pop(_k, None)
        st.info("ℹ️ Tenant 또는 실험이 변경되어 이전 Candidate가 초기화되었습니다.")
    st.session_state["qrels_ctx"] = _cur_ctx

    # ── 3. 문서 정보 ───────────────────────────────────────
    st.divider()
    st.subheader("📁 업로드된 문서")
    documents = registry.doc_repo.list_documents_by_tenant(tenant_id)
    if not documents:
        st.warning("업로드된 문서가 없습니다. 먼저 Document Upload 탭에서 문서를 업로드해주세요.")
        return

    doc_rows = []
    for doc in documents:
        ext = doc.file_name.rsplit(".", 1)[-1].upper() if "." in doc.file_name else "UNKNOWN"
        doc_rows.append({
            "문서명": doc.file_name,
            "문서 타입": ext,
            "Document ID": doc.document_id,
            "업로드 일시": str(doc.uploaded_at)[:19],
        })
    import pandas as pd
    st.dataframe(pd.DataFrame(doc_rows), use_container_width=True, hide_index=True)

    # ── 4. 질문 설정 ───────────────────────────────────────
    st.divider()
    st.subheader("❓ 질문 설정")

    questions = registry.question_repo.list_questions(tenant_id)
    q_options = [q.query_text for q in questions] + ["+ 새 질문 입력"]
    selected_q = st.selectbox("기존 질문 선택", q_options)

    query = ""
    question_id = None

    if selected_q == "+ 새 질문 입력":
        query = st.text_input("질문 입력", placeholder="예: '이 문서에서 보장하는 정기 점검 주기는 어떻게 되나요?'")
    else:
        query = selected_q
        for q in questions:
            if q.query_text == query:
                question_id = q.question_id
                break

    # ── 4. Candidate 생성 ──────────────────────────────────
    num_candidates = st.slider("후보군 개수 (Candidate size)", 5, 50, 20, key="qrels_num_cand_slider")

    RERANKER_MODELS = {
        "BAAI/bge-reranker-v2-m3":              "BGE Reranker v2-m3 ✅ 다국어 (한국어 지원)",
        "cross-encoder/ms-marco-MiniLM-L-6-v2": "MS-MARCO MiniLM ⚠️ 영어 전용",
    }
    reranker_model = st.selectbox(
        "🔀 Reranker 모델",
        options=list(RERANKER_MODELS.keys()),
        format_func=lambda x: RERANKER_MODELS[x],
        key="qrels_reranker_model"
    )
    if reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2":
        st.warning("⚠️ 영어 전용 모델입니다. 한국어 문서에 적용 시 Reranking 품질이 크게 저하될 수 있습니다. 한국어 문서에는 BGE Reranker v2-m3를 권장합니다.")

    if st.button("🔍 Candidate 생성 및 Reranking 실행", use_container_width=True):
        if not query:
            st.error("질문을 입력해주세요.")
            return

        if not question_id:
            question_id = str(uuid.uuid4())
            registry.question_repo.save_question(
                Question(question_id=question_id, experiment_id=experiment_id,
                         query_text=query, created_at=datetime.datetime.now())
            )

        fetch_k = num_candidates * 3

        with st.status("⏳ Pipeline 실행 중...", expanded=True) as status:

            # ── Step 1: Hybrid Retrieval ──────────────────────────
            s1 = st.empty()
            s1.write(f"🔍 **[1/4] Hybrid Retrieval** 실행 중 (Vector[{exp_model}] + BM25 RRF, fetch_k={fetch_k})...")
            try:
                retriever = registry.create_retriever("hybrid", exp_model)
                retrieved_candidates = retriever.retrieve_chunks(tenant_id, query, top_k=fetch_k)
                s1.write(f"✅ **[1/4] Hybrid Retrieval** 완료 — {len(retrieved_candidates)}개 검색됨")
            except Exception as e:
                s1.write(f"❌ **[1/4] Retrieval 실패**: {e}")
                status.update(label="오류 발생", state="error")
                st.stop()

            # ── Step 2: Candidate 필터링 ──────────────────────────
            s2 = st.empty()
            s2.write("🔧 **[2/4] Candidate 필터링** 중 (중복/노이즈 제거)...")
            try:
                filtered_candidates = registry.filter_chain.filter_candidates(retrieved_candidates)
                s2.write(f"✅ **[2/4] 필터링** 완료 — {len(filtered_candidates)}개 남음 ({len(retrieved_candidates) - len(filtered_candidates)}개 제거)")
            except Exception as e:
                s2.write(f"❌ **[2/4] 필터링 실패**: {e}")
                status.update(label="오류 발생", state="error")
                st.stop()

            # ── Step 3: Cross-Encoder Reranking ──────────────────
            s3 = st.empty()
            s3.write(f"🎯 **[3/4] Cross-Encoder Reranking** 중 (모델: {reranker_model.split('/')[-1]})...")
            try:
                reranker = registry.create_reranker(reranker_model)
                reranked_candidates = reranker.rerank(query, filtered_candidates, top_k=num_candidates)
                s3.write(f"✅ **[3/4] Reranking** 완료 — 상위 {len(reranked_candidates)}개 선정")
            except Exception as e:
                s3.write(f"❌ **[3/4] Reranking 실패**: {e}")
                status.update(label="오류 발생", state="error")
                st.stop()

            # ── Step 4: LLM Relevance Suggestion ─────────────────
            s4 = st.empty()
            s4.write(f"🤖 **[4/4] LLM Relevance Suggestion** 생성 중 ({len(reranked_candidates)}개 청크 평가)...")
            try:
                chunks_to_suggest = [c for c, _ in reranked_candidates]
                llm_scores_map = registry.suggester.suggest_scores(query, chunks_to_suggest)
                s4.write(f"✅ **[4/4] LLM Suggestion** 완료")
            except Exception as e:
                s4.write(f"⚠️ **[4/4] LLM Suggestion 실패** (점수 0으로 처리): {e}")
                llm_scores_map = {}

            # ── 결과 어셈블 ───────────────────────────────────────
            candidates = []
            for chunk, rerank_score in reranked_candidates:
                retriever_score = next(
                    (s for c, s in retrieved_candidates if c.chunk_id == chunk.chunk_id), 0.0
                )
                candidates.append({
                    "chunk": chunk,
                    "retriever_score": retriever_score,
                    "rerank_score": rerank_score,
                    "llm_suggestion": llm_scores_map.get(chunk.chunk_id, 0),
                })

            st.session_state['qrels_candidates']     = candidates
            st.session_state['current_query']        = query
            st.session_state['current_q_id']         = question_id
            st.session_state['current_exp_id']       = experiment_id
            st.session_state['qrels_num_candidates'] = num_candidates
            st.session_state['qrels_exp_model']      = exp_model
            st.session_state['llm_scores'] = {
                c['chunk'].chunk_id: c.get('llm_suggestion', 0)
                for c in candidates
            }
            for c in candidates:
                st.session_state[f"op_score_{c['chunk'].chunk_id}"] = None

            status.update(label=f"✅ Candidate {len(candidates)}개 생성 완료!", state="complete", expanded=False)

    # ── 5. HITL 검증 UI ────────────────────────────────────
    if 'qrels_candidates' in st.session_state:
        st.divider()
        st.subheader(f"📋 Candidate 검증: '{st.session_state['current_query']}'")

        candidates = st.session_state['qrels_candidates']

        if not candidates:
            st.warning("검색된 Candidate가 없습니다. 문서가 업로드되어 있는지 확인해주세요.")
            return

        LBL  = {0: "무관",     1: "부분 일치",  2: "정답"}
        ICON = {0: "🔴",       1: "🟡",          2: "🟢"}

        # 상태별 상단 배너 CSS
        BANNER = {
            None: ("background:#efefef;color:#888;border-left:5px solid #bbb;",
                   "⬜ 운영자 점수 미선택 — 아래에서 점수를 선택해주세요"),
            0:    ("background:#fde8e8;color:#c0392b;border-left:5px solid #e74c3c;",
                   "🔴 운영자 확정: 무관 (0점)"),
            1:    ("background:#fef9e7;color:#b7770d;border-left:5px solid #f39c12;",
                   "🟡 운영자 확정: 부분 일치 (1점)"),
            2:    ("background:#eafaf1;color:#1e8449;border-left:5px solid #27ae60;",
                   "🟢 운영자 확정: 정답 (2점)"),
        }

        # 진행률: 위젯 키에서 직접 읽기
        total = len(candidates)
        done  = sum(
            1 for c in candidates
            if st.session_state.get(f"op_score_{c['chunk'].chunk_id}") is not None
        )
        st.caption(f"진행률: {done} / {total} 완료")
        st.progress(done / total if total else 0)

        for i, c in enumerate(candidates):
            chunk        = c['chunk']
            rerank_score = c.get('rerank_score', 0.0)
            llm_score    = st.session_state['llm_scores'].get(chunk.chunk_id, 0)
            widget_key   = f"op_score_{chunk.chunk_id}"
            op_score_cur = st.session_state.get(widget_key)  # None or 0/1/2

            op_label = "⬜ 미선택" if op_score_cur is None else f"{ICON[op_score_cur]} {op_score_cur}점"
            with st.expander(
                f"Chunk {i+1}  |  Rerank: {rerank_score:.4f}  |  "
                f"LLM {ICON[llm_score]} {llm_score}점  |  운영자 {op_label}",
                expanded=True
            ):
                # ── 상태 배너 ──
                css, msg = BANNER[op_score_cur]
                st.markdown(
                    f'<div style="{css} padding:8px 12px; border-radius:4px; '
                    f'margin-bottom:10px; font-weight:600;">{msg}</div>',
                    unsafe_allow_html=True
                )

                st.write(chunk.text)
                st.caption(f"Page {chunk.page_number} | Chunk ID: {chunk.chunk_id}")

                st.divider()
                col_llm, col_op = st.columns(2)

                with col_llm:
                    st.markdown("**🤖 LLM 추천 점수**")
                    st.metric(label="LLM Score", value=f"{ICON[llm_score]} {LBL[llm_score]} ({llm_score}점)")

                with col_op:
                    st.markdown("**👤 운영자 확정 점수**")
                    # key로 session_state 직접 관리 — index 파라미터 사용 안 함
                    st.radio(
                        "점수 선택",
                        options=[0, 1, 2],
                        format_func=lambda x: f"{ICON[x]} {LBL[x]}",
                        key=widget_key,
                        horizontal=True
                    )

        # ── 6. Gold Qrels 저장 ─────────────────────────────
        st.divider()
        with st.expander("💾 Gold Qrels 데이터셋 저장"):
            current_query = st.session_state['current_query']
            auto_name     = current_query[:40] + ("..." if len(current_query) > 40 else "")
            dataset_name  = st.text_input("데이터셋 이름", value=auto_name)
            dataset_ver   = st.text_input("버전", value="v1.0")

            # 결정론적 dataset_id: question_id + version 조합 → UUID v5
            det_dataset_id = str(uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"{st.session_state['current_q_id']}_{dataset_ver}"
            ))
            st.caption(f"Dataset ID (결정론적): `{det_dataset_id}` — 같은 질문+버전은 항상 동일 ID로 upsert됩니다.")

            unscored = sum(
                1 for c in st.session_state['qrels_candidates']
                if st.session_state.get(f"op_score_{c['chunk'].chunk_id}") is None
            )
            if unscored > 0:
                st.warning(f"⚠️ 아직 {unscored}개의 Chunk가 미선택 상태입니다. 저장 시 0점(무관)으로 처리됩니다.")

            if st.button("최종 Gold Qrels 저장", type="primary", use_container_width=True):
                dataset_id = det_dataset_id
                registry.qrels_repo.save_dataset_version(DatasetVersion(
                    dataset_id=dataset_id,
                    experiment_id=st.session_state['current_exp_id'],
                    name=dataset_name,
                    version=dataset_ver,
                    chunk_config={"num_candidates": st.session_state.get('qrels_num_candidates', num_candidates)},
                    embedding_model=st.session_state.get('qrels_exp_model', exp_model),
                    created_at=datetime.datetime.now()
                ))

                saved_count    = 0
                llm_scores_map = st.session_state.get('llm_scores', {})
                for c in st.session_state['qrels_candidates']:
                    cid      = c['chunk'].chunk_id
                    op_score = st.session_state.get(f"op_score_{cid}")
                    registry.qrels_repo.save_qrel(GoldQrel(
                        dataset_id=dataset_id,
                        question_id=st.session_state['current_q_id'],
                        chunk_id=cid,
                        llm_score=llm_scores_map.get(cid, 0),
                        operator_score=op_score if op_score is not None else 0
                    ))
                    saved_count += 1

                st.success(f"총 {saved_count}개의 Qrels가 '{dataset_name} ({dataset_ver})'에 저장되었습니다!")
                del st.session_state['qrels_candidates']
                del st.session_state['llm_scores']
