from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import hashlib

# =========================================================
# Enterprise RAG 실험 플랫폼 도메인 모델 (13개 핵심 테이블 대응)
# =========================================================

# ----------------- 조직 계층 -----------------
@dataclass
class Tenant:
    tenant_id: str
    tenant_name: str
    created_at: datetime = field(default_factory=datetime.utcnow)

# ----------------- 실험 계층 -----------------
@dataclass
class Experiment:
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""
    name: str = ""
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ExperimentConfig:
    config_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str = ""
    retriever_type: str = "hybrid"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 500
    overlap: int = 50
    reranker_type: Optional[str] = None
    llm_model: str = "gpt-4-turbo"
    temperature: float = 0.0
    top_p: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ExperimentRun:
    """실행 단위. 특정 Config와 특정 데이터셋 버전을 결합해 언제 돌렸는지 기록."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    config_id: str = ""
    dataset_id: str = ""
    run_name: str = ""
    status: str = "running"
    started_at: datetime = field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None

# ----------------- 문서 계층 -----------------
@dataclass
class Document:
    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""
    file_name: str = ""
    file_hash: str = ""
    uploaded_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class DocumentVersion:
    doc_version_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    parser_name: str = "pdfplumber"
    parser_version: str = "0.10.3"
    chunk_size: int = 500
    overlap: int = 50
    embedding_model: str = "text-embedding-3-small"
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Chunk:
    chunk_id: str
    doc_version_id: str
    chunk_index: int
    page_number: int
    text: str
    token_count: int
    embedding_vector: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @staticmethod
    def generate_id(doc_version_id: str, chunk_index: int) -> str:
        return hashlib.sha256(f"{doc_version_id}_{chunk_index}".encode()).hexdigest()

# ----------------- 질문 계층 -----------------
@dataclass
class Question:
    question_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str = ""
    query_text: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

# ----------------- Gold Qrels 계층 (Evaluation Dataset) -----------------
@dataclass
class DatasetVersion:
    dataset_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str = ""
    name: str = "Initial Gold Set"
    version: str = "v1"
    chunk_config: Dict[str, Any] = field(default_factory=dict)
    embedding_model: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class GoldQrel:
    dataset_id: str
    question_id: str
    chunk_id: str
    llm_score: int        # LLM 자동 추천 점수 (0/1/2)
    operator_score: int   # 운영자 최종 확정 점수 (0/1/2)

# ----------------- 로그 & 평가 계층 -----------------
@dataclass
class RetrievalResult:
    run_id: str
    question_id: str
    chunk_id: str
    rank: int
    score: float

@dataclass
class GenerationResult:
    run_id: str
    question_id: str
    generated_answer: str
    latency_ms: float
    token_usage: int
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class EvaluationResult:
    run_id: str
    metric_name: str
    metric_value: float
    created_at: datetime = field(default_factory=datetime.utcnow)

# =========================================================
# 데이터 무결성 통합 객체 (Dataset)
# =========================================================

@dataclass
class Dataset:
    """
    RAG 실험 시스템 전역에서 사용하는 공통 데이터 모델 교환 포맷.
    핵심 데이터(Document, Chunk, Question, GoldQrel)를 하나의 논리적 덩어리로 묶습니다.
    """
    documents: List[Document] = field(default_factory=list)
    document_versions: List[DocumentVersion] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)
    questions: List[Question] = field(default_factory=list)
    gold_qrels: List[GoldQrel] = field(default_factory=list)
    
    @property
    def total_documents(self) -> int:
        return len(self.documents)
        
    @property
    def total_chunks(self) -> int:
        return len(self.chunks)
        
    @property
    def total_questions(self) -> int:
        return len(self.questions)
        
    @property
    def total_qrels(self) -> int:
        return len(self.gold_qrels)
