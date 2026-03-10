import re

class NoiseFilter:
    """노이즈 제거 (숫자만, 특수문자만, 등)"""
    
    def __init__(self, min_alpha_ratio=0.5):
        """min_alpha_ratio: 알파벳 비율 최소값"""
        self.min_alpha_ratio = min_alpha_ratio
    
    def check(self, chunk: str) -> bool:
        """노이즈 판단"""
        if not chunk or len(chunk.strip()) < 5:
            return False
        
        # 알파벳 비율 계산
        alpha_count = sum(1 for c in chunk if c.isalpha())
        alpha_ratio = alpha_count / len(chunk)
        
        if alpha_ratio < self.min_alpha_ratio:
            return False  # 노이즈 (숫자/기호만 많음)
        
        # URL, 이메일 등 완전 필터링 (선택)
        if re.search(r'(http|www|@)', chunk):
            return False
        
        return True