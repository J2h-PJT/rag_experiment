import streamlit as st
import pandas as pd
from typing import List, Optional, Tuple

from src.registry import Registry


def render_dashboard_tab():
    """Metrics Dashboard 탭 — Run 간 메트릭 비교 및 차트 시각화"""
    st.header("📊 Metrics Dashboard")
    st.markdown("여러 실험 Run의 메트릭을 비교하고 성능 추세를 분석합니다.")

    registry = Registry.get_instance()

    # ── 1. 사이드바: Tenant → Experiment → Dataset 선택 ──────
    tenant_id, experiment_id, dataset_id = _render_sidebar(registry)

    if not tenant_id or not experiment_id or not dataset_id:
        st.info("Tenant, Experiment, Dataset을 모두 선택해주세요.")
        return

    # ── 2. Dataset Runs 메트릭 조회 ────────────────────────
    try:
        raw_data = registry.run_repo.get_dataset_runs_metrics(dataset_id)
    except Exception as e:
        st.error(f"메트릭 조회 실패: {e}")
        return

    if not raw_data:
        st.info("이 Dataset의 완료된 Run이 없습니다. Experiment Runner에서 실험을 실행해주세요.")
        return

    # 데이터 정제 및 DataFrame 생성
    df = pd.DataFrame(raw_data)
    df = df[df["status"] == "COMPLETED"]  # 완료된 Run만 표시

    if df.empty:
        st.info("완료된 Run이 없습니다.")
        return

    # ── 3. Run 목록 테이블 ───────────────────────────────────
    st.divider()
    st.subheader("📋 Run 목록")
    _render_run_table(df)

    # ── 4. 메트릭 비교 차트 ──────────────────────────────────
    st.divider()
    st.subheader("📈 메트릭 비교 차트")
    _render_metric_charts(df)

    # ── 5. Run 상세 정보 ─────────────────────────────────────
    st.divider()
    st.subheader("🔍 Run 상세 정보")
    _render_run_detail(registry, df)


def _render_sidebar(registry: Registry) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """사이드바: Tenant → Experiment → Dataset 계층 선택"""
    st.sidebar.subheader("📋 데이터 선택")

    # Tenant 선택
    tenants = registry.doc_repo.list_tenants()
    if not tenants:
        st.warning("등록된 Tenant가 없습니다.")
        return None, None, None

    tenant_options = {t.tenant_name: t.tenant_id for t in tenants}
    selected_tenant_name = st.sidebar.selectbox("Tenant", list(tenant_options.keys()), key="dash_tenant_sel")
    tenant_id = tenant_options[selected_tenant_name]

    # 컨텍스트 변경 감지 (Tenant 변경 시 experiment/dataset 리셋)
    current_ctx = f"{tenant_id}"
    prev_ctx = st.session_state.get("dash_ctx_tenant")
    if current_ctx != prev_ctx:
        st.session_state["dash_experiment_id"] = None
        st.session_state["dash_dataset_id"] = None
        st.session_state["dash_ctx_tenant"] = current_ctx

    # Experiment 선택
    experiments = registry.experiment_repo.list_experiments(tenant_id)
    if not experiments:
        st.sidebar.warning("이 Tenant의 실험이 없습니다.")
        return tenant_id, None, None

    exp_options = {
        (e.get("name") or "<이름 없음>"): str(e["experiment_id"])
        for e in experiments
    }
    selected_exp_name = st.sidebar.selectbox("Experiment", list(exp_options.keys()), key="dash_exp_sel")
    experiment_id = exp_options[selected_exp_name]

    # 컨텍스트 변경 감지 (Experiment 변경 시 dataset 리셋)
    current_ctx = f"{tenant_id}|{experiment_id}"
    prev_ctx = st.session_state.get("dash_ctx_exp")
    if current_ctx != prev_ctx:
        st.session_state["dash_dataset_id"] = None
        st.session_state["dash_ctx_exp"] = current_ctx

    # Dataset 선택
    datasets = registry.qrels_repo.list_datasets_by_experiment(experiment_id)
    if not datasets:
        st.sidebar.warning("이 Experiment의 Dataset이 없습니다.")
        return tenant_id, experiment_id, None

    dataset_options = {f"{d.name} ({d.version})": d.dataset_id for d in datasets}
    selected_ds_label = st.sidebar.selectbox("Dataset", list(dataset_options.keys()), key="dash_ds_sel")
    dataset_id = dataset_options[selected_ds_label]

    return tenant_id, experiment_id, dataset_id


def _render_run_table(df: pd.DataFrame):
    """Run 목록을 DataFrame으로 표시"""
    # Run별 unique row (각 Run당 최대 1개만)
    run_summary = df.groupby("run_id").agg({
        "run_name": "first",
        "status": "first",
        "started_at": "first",
        "retriever_type": "first",
        "embedding_model": "first",
        "reranker_type": "first",
    }).reset_index()

    run_rows = []
    for _, row in run_summary.iterrows():
        # 이 Run의 모든 메트릭을 한 줄로 표시
        run_metrics = df[df["run_id"] == row["run_id"]][["metric_name", "metric_value"]].drop_duplicates()
        metric_str = " | ".join(
            f"{m['metric_name']}: {m['metric_value']:.4f}"
            for _, m in run_metrics.iterrows()
        )

        run_rows.append({
            "Run 이름": row["run_name"],
            "Retriever": row["retriever_type"] or "-",
            "Embedding": row["embedding_model"] or "-",
            "Reranker": row["reranker_type"] or "-",
            "시작": str(row["started_at"])[:19],
            "메트릭": metric_str,
        })

    st.dataframe(pd.DataFrame(run_rows), use_container_width=True, hide_index=True)


def _render_metric_charts(df: pd.DataFrame):
    """메트릭별 비교 차트 (Recall, MRR, NDCG)"""
    # Pivot: index=run_name, columns=metric_name, values=metric_value
    pivot = df.pivot_table(
        index="run_name",
        columns="metric_name",
        values="metric_value",
        aggfunc="first"
    )

    if pivot.empty:
        st.info("차트를 생성할 메트릭 데이터가 없습니다.")
        return

    # Metric 카테고리별 차트 생성
    metric_groups = {
        "Recall": [col for col in pivot.columns if col.startswith("Recall")],
        "MRR": [col for col in pivot.columns if col.startswith("MRR")],
        "NDCG": [col for col in pivot.columns if col.startswith("NDCG")],
    }

    for group_name, metrics in metric_groups.items():
        if metrics:
            st.markdown(f"**{group_name}**")
            st.bar_chart(pivot[metrics])
            st.markdown("")  # 여백


def _render_run_detail(registry: Registry, df: pd.DataFrame):
    """Run 상세 정보 및 복사용 요약 (expander)"""
    unique_runs = df["run_name"].unique()

    for run_name in unique_runs:
        run_data = df[df["run_name"] == run_name].iloc[0]
        run_id = run_data["run_id"]

        # Expander: Run 상세
        with st.expander(f"📌 {run_name}", expanded=False):
            # 기본 정보
            col1, col2, col3 = st.columns(3)
            col1.metric("Retriever", run_data["retriever_type"] or "-")
            col2.metric("Embedding", run_data["embedding_model"] or "-")
            col3.metric("Reranker", run_data["reranker_type"] or "-")

            # 모든 메트릭 카드
            st.markdown("**메트릭 요약**")
            run_metrics = df[df["run_id"] == run_id][["metric_name", "metric_value"]].drop_duplicates()
            metric_cols = st.columns(len(run_metrics))
            for col, (_, metric_row) in zip(metric_cols, run_metrics.iterrows()):
                col.metric(metric_row["metric_name"], f"{metric_row['metric_value']:.4f}")

            # 복사용 요약
            metric_str = "\n".join(
                f"  {m['metric_name']:<12}: {m['metric_value']:.4f}"
                for _, m in run_metrics.iterrows()
            )
            summary_text = f"""====== 메트릭 대시보드 — Run 요약 ======
Run        : {run_name}
실행 시간   : {str(run_data['started_at'])[:19]}
Retriever  : {run_data['retriever_type'] or '-'}
Embedding  : {run_data['embedding_model'] or '-'}
Reranker   : {run_data['reranker_type'] or '-'}

[메트릭]
{metric_str}
========================================="""

            with st.expander("📋 요약 복사", expanded=False):
                st.code(summary_text, language=None)
