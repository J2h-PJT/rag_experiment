import requests
from typing import List
from src.core.interfaces import IEmbedder

class OllamaEmbedder(IEmbedder):
    """
    로컬 Ollama를 이용한 임베딩 생성기.
    예: 모델명 "llama3.1:8b" 또는 "nomic-embed-text"
    """
    def __init__(self, model_name: str = "llama3.1:8b", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host
        # 임베딩 전용 엔드포인트
        self.api_url = f"{self.host}/api/embeddings"

    def get_model_name(self) -> str:
        return self.model_name
        
    def get_model_version(self) -> str:
        return "ollama-local"

    def embed_text(self, text: str) -> List[float]:
        """단일 텍스트를 Ollama에 요청하여 임베딩 추출"""
        payload = {
            "model": self.model_name,
            "prompt": text
        }
        try:
            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            # ollama는 'embedding' 키로 벡터 리턴
            return data.get("embedding", [])
        except Exception as e:
            print(f"Ollama Embedding Error for {self.model_name}: {e}")
            # 테스트/개발 단계 임시 조치: 오류 시 0벡터 반환
            # (pgvector 1536 등과 달라질 수 있으나 llama3.1의 임베딩 차원수 예외처리 필요)
            # 여기서는 Ollama 모델별 차원수 불일치 방지를 위해 빈 리스트 또는 예외 발생 권장
            raise e

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Ollama API는 배치 엔드포인트를 기본 지원하지 않을 수 있어 순차 호출"""
        return [self.embed_text(t) for t in texts]


class MockOpenAIEmbedder(IEmbedder):
    """
    OpenAI 호환 임베더 흉내 (API 키 미설정 환경 대상)
    실제 배포 시엔 openai 라이브러리로 대체.
    """
    def __init__(self, model_name="text-embedding-3-small"):
        self.model_name = model_name
        
    def get_model_name(self) -> str:
        return self.model_name
        
    def get_model_version(self) -> str:
        return "mock-1.0"
        
    def embed_text(self, text: str) -> List[float]:
        return [0.0] * 1536
        
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [[0.0] * 1536 for _ in texts]
