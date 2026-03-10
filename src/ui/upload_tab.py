import streamlit as st
import uuid
import datetime
from src.registry import Registry, AVAILABLE_EMBEDDING_MODELS
from src.core.models import Tenant

def render_upload_tab():
    st.header("📄 문서 데이터 인제스트")
    st.markdown("""
        평가 세트 구축을 위한 원천 문서를 업로드하고 청킹 및 임베딩 파이프라인을 실행합니다.
        동일한 파일은 해시를 통해 중복 업로드가 방지됩니다.
    """)

    registry = Registry.get_instance()

    # ── 1. Tenant 선택 / 생성 ──────────────────────────────
    st.subheader("🏢 Tenant 선택")

    tenants = registry.doc_repo.list_tenants()
    tenant_options = {t.tenant_name: t.tenant_id for t in tenants}
    display_options = list(tenant_options.keys()) + ["+ 새 Tenant 생성"]

    # 등록된 Tenant가 없으면 생성 폼을 바로 표시
    if not tenants:
        st.warning("등록된 Tenant가 없습니다. 새 Tenant를 생성해주세요.")
        next_id = "default_tenant_1"
    else:
        next_id = f"default_tenant_{len(tenants) + 1}"

    selected = st.selectbox("Tenant 선택", display_options, index=0 if tenants else len(display_options) - 1)

    tenant_id = None

    if selected == "+ 새 Tenant 생성" or not tenants:
        with st.form("new_tenant_form"):
            new_name = st.text_input("Tenant 이름", placeholder="예: 회사명 또는 팀명")
            new_id   = st.text_input("Tenant ID", value=next_id)
            submitted = st.form_submit_button("Tenant 생성")

        if submitted:
            if not new_name:
                st.error("Tenant 이름을 입력해주세요.")
                return
            t_id = new_id.strip() if new_id.strip() else next_id
            new_tenant = Tenant(
                tenant_id=t_id,
                tenant_name=new_name,
                created_at=datetime.datetime.utcnow()
            )
            registry.doc_repo.save_tenant(new_tenant)
            st.success(f"Tenant '{new_name}' (ID: {t_id}) 생성 완료!")
            st.rerun()
        return  # 생성 폼 표시 중에는 업로드 폼 숨김

    else:
        tenant_id = tenant_options[selected]
        st.info(f"선택된 Tenant ID: `{tenant_id}`")

    # ── 2. 문서 업로드 폼 ─────────────────────────────────
    st.divider()
    with st.form("upload_form"):
        uploaded_files = st.file_uploader("PDF 파일 선택", type=["pdf"], accept_multiple_files=True)

        st.subheader("⚙️ 전처리 설정")
        c1, c2 = st.columns(2)
        with c1:
            chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2000, value=500, step=100)
        with c2:
            chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=50, step=10)

        st.subheader("🧠 Embedding Model")
        model_keys = list(AVAILABLE_EMBEDDING_MODELS.keys())
        model_labels = [f"{k} — {v}" for k, v in AVAILABLE_EMBEDDING_MODELS.items()]
        selected_model_idx = st.selectbox(
            "임베딩 모델 선택",
            options=range(len(model_keys)),
            format_func=lambda i: model_labels[i],
            index=0
        )
        selected_model = model_keys[selected_model_idx]
        st.caption(f"선택된 모델: `{selected_model}`  |  ⚠️ 모델이 바뀌면 벡터 차원이 달라져 기존 문서와 혼용 불가")

        submit_button = st.form_submit_button("인제스트 실행", use_container_width=True)

    if submit_button:
        if not uploaded_files:
            st.error("파일을 선택해주세요.")
            return

        doc_manager = registry.create_doc_manager(selected_model)
        my_bar = st.progress(0, text="문서 처리 중...")
        results = []

        for idx, file in enumerate(uploaded_files):
            try:
                content = file.read()
                version_id = doc_manager.process_document(
                    tenant_id=tenant_id,
                    file_name=file.name,
                    file_content=content,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                results.append((file.name, "성공", selected_model, version_id))
            except Exception as e:
                results.append((file.name, "실패", selected_model, str(e)))

            my_bar.progress(
                (idx + 1) / len(uploaded_files),
                text=f"{file.name} 처리 완료 ({idx+1}/{len(uploaded_files)})"
            )

        st.success("처리가 완료되었습니다.")
        import pandas as pd
        st.dataframe(
            pd.DataFrame(results, columns=["파일명", "결과", "임베딩 모델", "Version ID / 오류"]),
            use_container_width=True, hide_index=True
        )
