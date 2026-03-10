from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from src.core.models import (
    GoldQrel, Document, DocumentVersion, Chunk, 
    Question, ExperimentConfig, Tenant, DatasetVersion
)

class ITenantRepository(ABC):
    @abstractmethod
    def save_tenant(self, tenant: Tenant) -> bool: pass
    
    @abstractmethod
    def get_tenant(self, tenant_id: str) -> Optional[Tenant]: pass

class IQuestionRepository(ABC):
    @abstractmethod
    def save_question(self, question: Question) -> bool: pass
    
    @abstractmethod
    def get_question(self, tenant_id: str, question_id: str) -> Optional[Question]: pass
    
    @abstractmethod
    def list_questions(self, tenant_id: str) -> List[Question]: pass

class IQrelsRepository(ABC):
    @abstractmethod
    def save_dataset_version(self, version: DatasetVersion) -> bool: pass

    @abstractmethod
    def get_dataset_version(self, dataset_id: str) -> Optional[DatasetVersion]: pass

    @abstractmethod
    def save_qrel(self, record: GoldQrel) -> bool: pass
    
    @abstractmethod
    def get_qrels_by_dataset(self, dataset_id: str) -> List[GoldQrel]: pass
    
    @abstractmethod
    def get_qrel_for_question(self, dataset_id: str, question_id: str) -> List[GoldQrel]: pass

class IDocumentRepository(ABC):
    @abstractmethod
    def save_document(self, doc: Document) -> bool: pass
    
    @abstractmethod
    def get_document_by_hash(self, tenant_id: str, file_hash: str) -> Optional[Document]: pass

    @abstractmethod
    def save_document_version(self, version: DocumentVersion) -> bool: pass

    @abstractmethod
    def get_latest_version(self, document_id: str) -> Optional[DocumentVersion]: pass
    
    @abstractmethod
    def save_chunks(self, chunks: List[Chunk]) -> bool: pass

    @abstractmethod
    def get_chunks_by_version(self, doc_version_id: str) -> List[Chunk]: pass

    @abstractmethod
    def search_chunks_by_embedding(self, tenant_id: str, embedding: List[float], embedding_model: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """임베딩 벡터 기반 유사도 검색 (Cosine Distance). 동일 embedding_model 청크만 검색."""
        pass

    @abstractmethod
    def list_all_chunks_by_tenant(self, tenant_id: str) -> List[Chunk]:
        """특정 테넌트의 모든 문서 버전에 속한 청크들을 반환 (BM25 인덱싱 등 용도)"""
        pass

class IExperimentRepository(ABC):
    @abstractmethod
    def save_config(self, config: ExperimentConfig) -> bool: pass
    
    @abstractmethod
    def get_config(self, tenant_id: str, experiment_id: str) -> Optional[ExperimentConfig]: pass

    @abstractmethod
    def list_experiments(self, tenant_id: str) -> List[dict]: pass

    @abstractmethod
    def create_experiment(self, experiment_id: str, tenant_id: str, experiment_name: str, experiment_description: str) -> bool: pass


# ==========================================
# 기능 인터페이스 (Extensibility / Swappability)
# ==========================================

class IRetriever(ABC):
    """
    RAG 검색 알고리즘 추상화 인터페이스.
    실험 시 Vector Search, BM25, Hybrid 등을 자유롭게 교체하여
    검색 성능(Recall, MRR 등)을 평가하기 위함입니다.
    """
    @abstractmethod
    def retrieve_chunks(self, tenant_id: str, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """검색 쿼리 문자열에 대하여 청크와 점수/거리 값을 반환"""
        pass

class IEmbedder(ABC):
    """
    임베딩 모델을 추상화하는 인터페이스.
    실험 시 OpenAI, Ollama(llama3.1:8b 등), HuggingFace 모델을
    자유롭게 교체(Swap)하여 테스트하기 위함입니다.
    """
    @abstractmethod
    def get_model_name(self) -> str:
        """모델명 반환 (예: 'text-embedding-3-small', 'nomic-embed-text')"""
        pass
        
    @abstractmethod
    def get_model_version(self) -> str:
        """모델 버전 반환"""
        pass
        
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """단일 텍스트 임베딩"""
        pass
        
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """복수 텍스트 배치 임베딩 (최적화용)"""
        pass

class ILLMGenerator(ABC):
    """
    LLM 추론 모델을 추상화하는 인터페이스.
    실험 시 OpenAI(gpt-4), Ollama(llama3.1:8b) 등 모델을 
    교체해가며 RAG의 답변 퀄리티 피드백을 수집하기 위함입니다.
    """
    @abstractmethod
    def get_model_name(self) -> str:
        pass
        
    @abstractmethod
    def generate_answer(self, prompt: str, temperature: float = 0.0) -> str:
        """단건 프롬프트에 대한 답변 문자열 반환"""
        pass
