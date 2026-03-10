import requests
from src.core.interfaces import ILLMGenerator

class OllamaGenerator(ILLMGenerator):
    """
    로컬 Ollama를 이용한 LLM 생성기.
    예: 모델명 "llama3.1:8b" 
    """
    def __init__(self, model_name: str = "llama3.1:8b", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host
        self.api_url = f"{self.host}/api/generate"

    def get_model_name(self) -> str:
        return self.model_name

    def generate_answer(self, prompt: str, temperature: float = 0.0) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except Exception as e:
            print(f"Ollama Generation Error for {self.model_name}: {e}")
            return f"[Error generating answer with {self.model_name}]"

class MockOpenAIGenerator(ILLMGenerator):
    """
    OpenAI 호환 텍스트 제너레이터 흉내 (API 키 미설정 환경 대상).
    """
    def __init__(self, model_name="gpt-4-turbo"):
        self.model_name = model_name

    def get_model_name(self) -> str:
        return self.model_name

    def generate_answer(self, prompt: str, temperature: float = 0.0) -> str:
        return f"이것은 {self.model_name}의 Mock 응답입니다. (Temperature: {temperature})\n프롬프트 길이: {len(prompt)}"
