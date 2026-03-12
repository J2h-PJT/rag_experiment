import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import DictCursor
from typing import List, Optional
import os
from contextlib import contextmanager

# SQLAlchemy 스타일의 URI도 해석할 수 있도록 전처리 함수 추가
def clean_connection_url(url: str) -> str:
    """psycopg2가 직접 호출될 수 있도록 postgresql+psycopg2:// 부분을 postgresql:// 로 변경합니다."""
    if url and url.startswith("postgresql+psycopg2://"):
        return url.replace("postgresql+psycopg2://", "postgresql://", 1)
    return url

class DatabaseManager:
    """
    Enterprise RAG 전체 생태계를 지탱하는 Database 관리자.
    (13개 핵심 테이블의 생성 및 세션 라이프사이클 관리)
    Connection Pool을 활용하여 커넥션 재사용성을 보장합니다.
    """
    
    def __init__(self, connection_url: Optional[str] = None, minconn: int = 1, maxconn: int = 20):
        # 기본값으로 사용자가 입력한 SQLAlchemy 형태의 주소 세팅
        env_url = os.environ.get("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5433/rag_db")
        raw_url = connection_url or env_url
        self.connection_url = clean_connection_url(raw_url)
        
        # 스레드 안전한 커넥션 풀 초기화
        self._pool = ThreadedConnectionPool(minconn, maxconn, dsn=self.connection_url)
    
    @contextmanager
    def get_connection(self):
        """커넥션 풀에서 커넥션을 대여하고 사용 후 반납합니다."""
        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            self._pool.putconn(conn)
    
    def close(self):
        """애플리케이션 종료 시 커넥션 풀에 속한 모든 커넥션을 닫습니다."""
        if self._pool:
            self._pool.closeall()

    @contextmanager
    def transaction(self):
        """자동 Commit 및 예외 발생 시 Rollback을 지원하는 트랜잭션 블록"""
        with self.get_connection() as conn:
            try:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    yield cur
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e

    def initialize_schemas(self):
        with self.transaction() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

            # ----------------- 4.1. ~ 4.4. 조직, 실험, 설정, 실행 -----------------
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tenants (
                    tenant_id VARCHAR(100) PRIMARY KEY,
                    tenant_name VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id UUID PRIMARY KEY,
                    tenant_id VARCHAR(100) REFERENCES tenants(tenant_id),
                    name VARCHAR(255),
                    description TEXT,
                    embedding_model VARCHAR(100) DEFAULT 'bge-m3',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # 기존 테이블에 embedding_model 컬럼 없으면 추가
            cur.execute("""
                ALTER TABLE experiments ADD COLUMN IF NOT EXISTS embedding_model VARCHAR(100) DEFAULT 'bge-m3';
            """)
            
            # 재현성의 핵심: 파라미터 불변 세트
            cur.execute("""
                CREATE TABLE IF NOT EXISTS experiment_configs (
                    config_id UUID PRIMARY KEY,
                    experiment_id UUID REFERENCES experiments(experiment_id) ON DELETE CASCADE,
                    retriever_type VARCHAR(50),
                    embedding_model VARCHAR(100),
                    chunk_size INT,
                    overlap INT,
                    reranker_type VARCHAR(100),
                    llm_model VARCHAR(100),
                    temperature FLOAT,
                    top_p FLOAT,
                    fusion_weight FLOAT DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # ----------------- 4.5. ~ 4.7. 문서 파이프라인 -----------------
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id UUID PRIMARY KEY,
                    tenant_id VARCHAR(100) REFERENCES tenants(tenant_id),
                    file_name VARCHAR(255),
                    file_hash VARCHAR(64) UNIQUE,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_versions (
                    doc_version_id UUID PRIMARY KEY,
                    document_id UUID REFERENCES documents(document_id) ON DELETE CASCADE,
                    parser_name VARCHAR(100),
                    parser_version VARCHAR(50),
                    chunk_size INT,
                    overlap INT,
                    embedding_model VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id VARCHAR(64) PRIMARY KEY,
                    doc_version_id UUID REFERENCES document_versions(doc_version_id) ON DELETE CASCADE,
                    chunk_index INT,
                    page_number INT,
                    text TEXT,
                    token_count INT,
                    embedding_vector vector,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # 기존 vector(1024) 컬럼을 차원 없는 vector로 마이그레이션
            cur.execute("""
                ALTER TABLE chunks ALTER COLUMN embedding_vector TYPE vector
                USING embedding_vector::vector;
            """)
            
            # ----------------- 4.8. ~ 4.10. 질문 및 평가 정답 -----------------
            cur.execute("""
                CREATE TABLE IF NOT EXISTS questions (
                    question_id UUID PRIMARY KEY,
                    experiment_id UUID REFERENCES experiments(experiment_id) ON DELETE CASCADE,
                    query_text TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 정답 세트 묶음의 버저닝 (Dataset Versioning)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS dataset_versions (
                    dataset_id UUID PRIMARY KEY,
                    experiment_id UUID REFERENCES experiments(experiment_id) ON DELETE CASCADE,
                    name VARCHAR(255),
                    version VARCHAR(50),
                    chunk_config JSONB, -- {chunk_size, overlap, etc}
                    embedding_model VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 세트에 포함된 질문-청크 매핑
            cur.execute("""
                CREATE TABLE IF NOT EXISTS gold_qrels (
                    dataset_id UUID REFERENCES dataset_versions(dataset_id) ON DELETE CASCADE,
                    question_id UUID REFERENCES questions(question_id) ON DELETE CASCADE,
                    chunk_id VARCHAR(64) REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                    llm_score INT NOT NULL DEFAULT 0,
                    operator_score INT NOT NULL DEFAULT 0,
                    PRIMARY KEY (dataset_id, question_id, chunk_id)
                );
            """)
            # 기존 테이블 마이그레이션
            cur.execute("ALTER TABLE gold_qrels ADD COLUMN IF NOT EXISTS llm_score INT NOT NULL DEFAULT 0;")
            cur.execute("ALTER TABLE gold_qrels ADD COLUMN IF NOT EXISTS operator_score INT NOT NULL DEFAULT 0;")
            # 구 스키마 정리: relevance_score 컬럼 제거
            cur.execute("ALTER TABLE gold_qrels DROP COLUMN IF EXISTS relevance_score;")
            cur.execute("ALTER TABLE experiment_configs ADD COLUMN IF NOT EXISTS fusion_weight FLOAT DEFAULT 1.0;")

            # ----------------- 4.4. 실제 실행 단위 (Runs) -----------------
            cur.execute("""
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    run_id UUID PRIMARY KEY,
                    config_id UUID REFERENCES experiment_configs(config_id) ON DELETE CASCADE,
                    dataset_id UUID REFERENCES dataset_versions(dataset_id) ON DELETE RESTRICT,
                    run_name VARCHAR(255),
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    finished_at TIMESTAMP,
                    status VARCHAR(50) DEFAULT 'RUNNING'
                );
            """)

            # ----------------- 4.11. ~ 4.13. 실행 로그 & 메트릭 -----------------
            cur.execute("""
                CREATE TABLE IF NOT EXISTS retrieval_results (
                    run_id UUID REFERENCES experiment_runs(run_id) ON DELETE CASCADE,
                    question_id UUID REFERENCES questions(question_id) ON DELETE CASCADE,
                    chunk_id VARCHAR(64) REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                    rank INT,
                    score FLOAT,
                    PRIMARY KEY (run_id, question_id, chunk_id)
                );
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS generation_results (
                    run_id UUID REFERENCES experiment_runs(run_id) ON DELETE CASCADE,
                    question_id UUID REFERENCES questions(question_id) ON DELETE CASCADE,
                    generated_answer TEXT,
                    latency_ms FLOAT,
                    token_usage INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (run_id, question_id)
                );
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    run_id UUID REFERENCES experiment_runs(run_id) ON DELETE CASCADE,
                    metric_name VARCHAR(100),
                    metric_value FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (run_id, metric_name)
                );
            """)
