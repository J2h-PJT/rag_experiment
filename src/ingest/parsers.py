import io
import pdfplumber
from typing import List, Dict, Any

class RealPDFExtractor:
    """
    실제 pdfplumber를 이용한 PDF 텍스트 및 페이지 추출기.
    """
    def __init__(self):
        self.parser_name = "pdfplumber"
        self.parser_version = pdfplumber.__version__

    def extract_text_by_page(self, file_bytes: bytes) -> List[Dict[str, Any]]:
        """
        PDF를 읽어 페이지 단위로 텍스트를 추출합니다.
        반환: [{"page_number": 1, "text": "...", ...}, ...]
        """
        pages_content = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages_content.append({
                    "page_number": i + 1,
                    "text": text
                })
        return pages_content


from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

class RealChunker:
    """
    LangChain의 RecursiveCharacterTextSplitter 적용.
    LLM/임베딩 토큰 카운터로 tiktoken 사용 (또는 Llama3 토크나이저).
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, encoding_name: str = "cl100k_base"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 토큰 카운팅 용도 (gpt-4 용 cl100k_base, llama3용은 필요 시 교체)
        self.encoding = tiktoken.get_encoding(encoding_name)
        
        # 문자 단위지만 의미적(문단, 문장 단어)으로 자연스럽게 잘라주는 랭체인 스플리터
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def chunk_page(self, text: str, page_number: int, start_idx: int = 0) -> List[Dict[str, Any]]:
        """
        단일 페이지의 텍스트를 받아 청크 단위로 자르고 토큰수를 계산하여 리턴합니다.
        """
        if not text.strip():
            return []
            
        chunks_str = self.splitter.split_text(text)
        result = []
        
        for idx, chunk_str in enumerate(chunks_str):
            result.append({
                "page_num": page_number,
                "text": chunk_str,
                "token_count": self.count_tokens(chunk_str)
            })
            
        return result
