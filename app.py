import streamlit as st
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# UI 컴포넌트 임포트
from src.ui.upload_tab import render_upload_tab
from src.ui.qrels_tab import render_qrels_tab
from src.ui.runner_tab import render_runner_tab
from src.ui.dashboard_tab import render_dashboard_tab

def main():
    st.set_page_config(
        page_title="Enterprise RAG Evaluation Platform",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🧪 Enterprise RAG Evaluation Platform")
    st.caption("Advanced Experiment Management & Gold Qrels Generation for High-Quality RAG Systems")

    # 사이드바 설정
    st.sidebar.image("https://img.icons8.com/isometric/100/null/experimental-research-isometric.png", width=100)
    st.sidebar.header("Navigation")
    
    app_mode = st.sidebar.radio(
        "기능 선택",
        ["Document Upload", "Gold Qrels Builder", "Experiment Runner", "Metrics Dashboard"]
    )

    st.sidebar.divider()
    st.sidebar.info("""
    **V1.0.0 (Admin Edition)**
    Developed for Advanced Agentic Coding
    """)

    # 모드에 따른 탭 렌더링
    if app_mode == "Document Upload":
        render_upload_tab()
    elif app_mode == "Gold Qrels Builder":
        render_qrels_tab()
    elif app_mode == "Experiment Runner":
        render_runner_tab()
    elif app_mode == "Metrics Dashboard":
        render_dashboard_tab()

if __name__ == "__main__":
    main()
