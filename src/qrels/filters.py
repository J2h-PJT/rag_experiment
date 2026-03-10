from typing import List, Tuple, Set
from src.core.models import Chunk

class CandidateFilter:
    """
    Candidate Filtering 파이프라인.
    Gold Dataset 품질을 결정짓는 핵심 모듈.
    """
    def __init__(self, remove_duplicates: bool = True, noise_threshold: int = 50):
        self.remove_duplicates = remove_duplicates
        self.noise_threshold = noise_threshold

    def filter_candidates(self, candidates: List[Tuple[Chunk, float]]) -> List[Tuple[Chunk, float]]:
        filtered = candidates
        
        if self.remove_duplicates:
            filtered = self._apply_duplicate_filter(filtered)
            
        filtered = self._apply_noise_filter(filtered)
        
        # 향후 Section Filter 로직 추가 가능 (특정 목차, 부록 등의 Chunk 배제)
        # filtered = self._apply_section_filter(filtered)
        
        return filtered

    def _apply_duplicate_filter(self, candidates: List[Tuple[Chunk, float]]) -> List[Tuple[Chunk, float]]:
        """동일한 텍스트 내용을 가지는 Chunk를 Hash/Set 기반으로 제거합니다."""
        unique_texts: Set[str] = set()
        filtered: List[Tuple[Chunk, float]] = []
        
        for chunk, score in candidates:
            # 공백을 제거한 텍스트를 고유 키로 활용 (단순 해싱 목적)
            normalized_text = "".join(chunk.text.split())
            if normalized_text not in unique_texts:
                unique_texts.add(normalized_text)
                filtered.append((chunk, score))
                
        return filtered

    def _apply_noise_filter(self, candidates: List[Tuple[Chunk, float]]) -> List[Tuple[Chunk, float]]:
        """Header, Footer 또는 너무 짧은 노이즈 성 청크를 걸러냅니다."""
        filtered: List[Tuple[Chunk, float]] = []
        for chunk, score in candidates:
            # 예: 토큰 길이가 너무 적으면 노이즈로 간주
            if chunk.token_count >= self.noise_threshold:
                filtered.append((chunk, score))
        return filtered
