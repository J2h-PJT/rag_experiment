import re
from typing import List

class TextPreprocessor:
    """PDF 텍스트 정제 및 전처리"""

    @staticmethod
    def clean_text(text: str) -> str:
        """텍스트 정제 — 블랙리스트 방식으로 최소한만 제거"""
        # 줄 바꿈 정규화: 3개 이상 연속 줄바꿈 → 2개로
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 같은 줄 내 과도한 공백만 축소 (줄바꿈은 보존)
        text = re.sub(r'[^\S\n]+', ' ', text)

        # 양쪽 공백 제거
        text = text.strip()

        return text

    @staticmethod
    def remove_boilerplate(text: str) -> str:
        """헤더/푸터/보일러플레이트 제거 — 숫자만 있는 줄(페이지 번호) 위주로 제거"""
        lines = text.split('\n')

        filtered_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                filtered_lines.append(line)  # 빈 줄은 단락 구분자로 보존
                continue

            # 숫자만 있는 줄(페이지 번호) 제거
            if re.fullmatch(r'\d+', stripped):
                continue

            # 3자 미만 줄만 제거 (섹션 제목 등 짧은 의미 단위는 보존)
            if len(stripped) < 3:
                continue

            filtered_lines.append(line)

        return '\n'.join(filtered_lines)

    @staticmethod
    def preprocess(text: str) -> str:
        """전체 전처리 파이프라인"""
        text = TextPreprocessor.clean_text(text)
        text = TextPreprocessor.remove_boilerplate(text)
        return text