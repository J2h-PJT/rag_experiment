import os
from src.db.database_manager import DatabaseManager
from src.db.postgres_repository import (
    PostgresDocumentRepository,
    PostgresQrelsRepository,
    PostgresQuestionRepository,
    PostgresExperimentRepository,
    PostgresRunRepository,
)
from src.evaluation.runner import ExperimentRunner
from src.qrels.gold_qrels_engine import GoldQrelsEngine
from src.qrels.retrievers import VectorRetriever, BM25Retriever
from src.qrels.hybrid_retriever import HybridRetriever
from src.qrels.filters import CandidateFilter
from src.qrels.reranker import CrossEncoderReranker
from src.qrels.llm_suggester import LLMSuggester
from src.core.document_manager import DocumentManager
from src.ingest.embedders import OllamaEmbedder, MockOpenAIEmbedder
from src.llm.generator import OllamaGenerator

# 사용 가능한 임베딩 모델 목록 (모델명: 설명)
AVAILABLE_EMBEDDING_MODELS = {
    "bge-m3":            "BGE-M3 (Ollama) — 다국어, 고성능 권장",
    "nomic-embed-text":  "Nomic Embed Text (Ollama) — 경량, 768dim",
    "mxbai-embed-large": "MixedBread Large (Ollama) — 영문 특화",
    "mock":              "Mock Embedder — 테스트용 (0벡터)",
}

class Registry:
    """글로벌 서비스 인스턴스를 관리하는 레지스트리"""
    _instance = None

    def __init__(self):
        db_url = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5433/rag_db")
        self.db_manager = DatabaseManager(db_url)
        self.db_manager.initialize_schemas()

        # Repositories
        self.doc_repo = PostgresDocumentRepository(self.db_manager)
        self.qrels_repo = PostgresQrelsRepository(self.db_manager)
        self.question_repo = PostgresQuestionRepository(self.db_manager)
        self.experiment_repo = PostgresExperimentRepository(self.db_manager)
        self.run_repo = PostgresRunRepository(self.db_manager)

        # Components
        # 임시 임베딩 사용 (Ollama가 설치되어 있다면)
        self.embedder = OllamaEmbedder(model_name="bge-m3")  # 또는 "llama3.1:8b"nomic-embed-text(dimension=768)
        
        # 테스트용 Mock (임베딩 서버가 없을 때)
        # self.embedder = MockOpenAIEmbedder()
        
        self.llm = OllamaGenerator(model_name="llama3.1:8b")
        
        # Retrievers
        self.vector_retriever = VectorRetriever(self.doc_repo, self.embedder)
        self.bm25_retriever = BM25Retriever(self.doc_repo)
        self.hybrid_retriever = HybridRetriever([self.vector_retriever, self.bm25_retriever])
        
        # Pipelines
        self.filter_chain = CandidateFilter()
        self.reranker = CrossEncoderReranker() # 내부에서 모델 로드
        self.suggester = LLMSuggester(self.llm)
        
        # Core Engine
        self.engine = GoldQrelsEngine(
            retriever=self.hybrid_retriever,
            filter_chain=self.filter_chain,
            reranker=self.reranker,
            suggester=self.suggester
        )

        # Ingestion Service (기본 embedder 사용)
        self.doc_manager = DocumentManager(self.doc_repo, self.embedder)

    def create_embedder(self, model_name: str) -> "IEmbedder":
        """선택한 모델명으로 임베더 인스턴스를 생성합니다."""
        if model_name == "mock":
            return MockOpenAIEmbedder()
        return OllamaEmbedder(model_name=model_name)

    def create_doc_manager(self, model_name: str) -> DocumentManager:
        """선택한 임베딩 모델로 DocumentManager를 생성합니다."""
        embedder = self.create_embedder(model_name)
        return DocumentManager(self.doc_repo, embedder)

    def create_retriever(self, retriever_type: str, embedding_model: str,
                         vector_weight: float = 0.5, bm25_weight: float = 0.5):
        """retriever_type(hybrid/vector/bm25)과 embedding_model로 Retriever 생성"""
        embedder = self.create_embedder(embedding_model)
        vector_r  = VectorRetriever(self.doc_repo, embedder)
        bm25_r    = BM25Retriever(self.doc_repo)
        if retriever_type == "vector":
            return vector_r
        elif retriever_type == "bm25":
            return bm25_r
        return HybridRetriever([vector_r, bm25_r], weights=[vector_weight, bm25_weight])

    def create_reranker(self, model_name: str) -> CrossEncoderReranker:
        """선택한 모델명으로 Reranker 인스턴스를 생성합니다."""
        return CrossEncoderReranker(model_name=model_name)

    def create_experiment_runner(self, reranker_model: str = None) -> ExperimentRunner:
        reranker = self.create_reranker(reranker_model) if reranker_model else self.reranker
        return ExperimentRunner(
            run_repo=self.run_repo,
            qrels_repo=self.qrels_repo,
            question_repo=self.question_repo,
            llm=self.llm,
            reranker=reranker,
        )

    def create_engine_for_model(self, embedding_model: str, reranker_model: str = None) -> GoldQrelsEngine:
        """실험의 embedding_model에 맞는 검색 엔진을 생성합니다."""
        embedder = self.create_embedder(embedding_model)
        vector_retriever = VectorRetriever(self.doc_repo, embedder)
        bm25_retriever   = BM25Retriever(self.doc_repo)
        hybrid_retriever = HybridRetriever([vector_retriever, bm25_retriever])
        reranker = self.create_reranker(reranker_model) if reranker_model else self.reranker
        return GoldQrelsEngine(
            retriever=hybrid_retriever,
            filter_chain=self.filter_chain,
            reranker=reranker,
            suggester=self.suggester
        )

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
