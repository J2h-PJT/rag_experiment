import re
from typing import List

class TextPreprocessor:
    """PDF 텍스트 정제 및 전처리"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """텍스트 정제"""
        # 과도한 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 특수문자 정제 (필요에 따라)
        text = re.sub(r'[^\w\s\.\,\!\?\-\(\)\[\]\'\"]', '', text)
        
        # 양쪽 공백 제거
        text = text.strip()
        
        return text
    
    @staticmethod
    def remove_boilerplate(text: str) -> str:
        """헤더/푸터/보일러플레이트 제거"""
        lines = text.split('\n')
        
        # 너무 짧은 라인 필터링 (페이지 번호, 날짜 등)
        filtered_lines = [
            line for line in lines 
            if len(line.strip()) > 10
        ]
        
        return '\n'.join(filtered_lines)
    
    @staticmethod
    def preprocess(text: str) -> str:
        """전체 전처리 파이프라인"""
        text = TextPreprocessor.clean_text(text)
        text = TextPreprocessor.remove_boilerplate(text)
        return text