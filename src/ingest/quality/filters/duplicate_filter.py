from typing import List

class DuplicateFilter:
    """중복 청크 제거"""
    
    def __init__(self):
        self.seen_texts = set()
    
    def check(self, chunk: str) -> bool:
        """해시 기반 중복 검사"""
        chunk_hash = hash(chunk.strip().lower())
        
        if chunk_hash in self.seen_texts:
            return False  # 중복
        
        self.seen_texts.add(chunk_hash)
        return True  # 새로운 텍스트
    
    def reset(self):
        """문서별로 초기화"""
        self.seen_texts.clear()