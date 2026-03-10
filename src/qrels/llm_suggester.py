from typing import List, Tuple, Dict
from src.core.models import Chunk
from src.core.interfaces import ILLMGenerator
import re

class LLMSuggester:
    """
    LLM Relevance Suggestion
    관리자의 데이터셋 구성 수고를 덜기 위해 LLM을 사용하여 
    각 Chunk의 Relevance (0: irrelevant, 1: partially, 2: highly)를 추천합니다.
    """
    def __init__(self, llm_generator: ILLMGenerator):
        self.llm = llm_generator
        self.prompt_template = """
You are an expert evaluator for Enterprise RAG systems.
Given the QUESTION, evaluate if the provided DOCUMENT_CHUNK contains the exact answer.

Score Criteria:
0 = Irrelevant (Does not contain information to answer the question)
1 = Partially relevant (Contains some related information but not the complete or exact answer)
2 = Highly relevant (Contains the exact and complete answer to the question)

Return ONLY a single integer (0, 1, or 2). Do not include any explanation or extra text.

QUESTION: {query}
DOCUMENT_CHUNK: {chunk_text}

Score:
"""

    def suggest_scores(self, query: str, chunks: List[Chunk]) -> Dict[str, int]:
        """
        주어진 Chunk 리스트에 대해 LLM에게 Relevance 판단을 요청하여 매핑된 딕셔너리를 반환합니다.
        Returns: Dict[chunk_id -> score(0,1,2)]
        """
        suggestions: Dict[str, int] = {}
        for chunk in chunks:
            prompt = self.prompt_template.strip().format(query=query, chunk_text=chunk.text)
            
            # 온도 0.0 으로 가장 결정론적(Deterministic)인 단일 토큰 답변 유도
            try:
                response = self.llm.generate_answer(prompt, temperature=0.0)
                score = self._parse_score(response)
                suggestions[chunk.chunk_id] = score
            except Exception as e:
                print(f"[LLMSuggester Error] Chunk {chunk.chunk_id} 추론 실패: {e}")
                suggestions[chunk.chunk_id] = 0 # 실패 시 안전하게 0점 처리
                
        return suggestions

    def _parse_score(self, llm_output: str) -> int:
        """
        답변 텍스트에서 0, 1, 2 숫자를 단 하나만 안전하게 파싱합니다.
        가끔 LLM 필터나 프롬프트 지시를 무시하고 긴 텍스트가 올 수 있기 때문.
        """
        # 정규표현식으로 맨 처음 나오는 숫자를 추출 (1자리 숫자에 한함)
        match = re.search(r'\b([012])\b', llm_output)
        if match:
            return int(match.group(1))
        
        # 대체 룰 (숫자로 변환 가능한지 완전 하드코딩 검증)
        text = llm_output.strip()
        if "2" in text: return 2
        if "1" in text: return 1
        return 0
