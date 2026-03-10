from typing import List
from src.ingest.quality.filters import LengthFilter, DuplicateFilter, NoiseFilter

class QualityPipeline:
    """청크 품질 검증 파이프라인"""
    
    def __init__(self, min_tokens: int = 20, min_alpha_ratio: float = 0.5):
        self.filters = [
            DuplicateFilter(),
            NoiseFilter(min_alpha_ratio=min_alpha_ratio),
            LengthFilter(min_tokens=min_tokens)
        ]
    
    def filter_chunks(self, chunks: List[str]) -> List[str]:
        """모든 필터를 순차적으로 적용"""
        filtered = []
        
        for chunk in chunks:
            passed = True
            for filter_obj in self.filters:
                if not filter_obj.check(chunk):
                    passed = False
                    break
            
            if passed:
                filtered.append(chunk)
        
        return filtered
    
    def reset(self):
        """문서 전환 시 필터 초기화"""
        for f in self.filters:
            if hasattr(f, 'reset'):
                f.reset()