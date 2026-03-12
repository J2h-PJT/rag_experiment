import re
import uuid
import datetime
from typing import List, Optional
from src.core.models import Document, DocumentVersion, Chunk
from src.core.interfaces import IDocumentRepository, IEmbedder
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber

def _is_two_column_page(page) -> bool:
    """
    페이지가 좌우 2-컬럼 레이아웃인지 자동 감지.
    감지 기준:
      1) 가로가 세로보다 넓은 페이지 (landscape)
      2) 페이지 왼쪽 40% 이내에서 시작하는 단어가 전체의 25% 이상
      3) 페이지 오른쪽 40% 이내에서 시작하는 단어가 전체의 25% 이상
    → 컬럼 사이 공백이 없이 밀착된 2-컬럼도 감지 가능
    """
    if page.width <= page.height:
        return False  # 세로 방향 PDF는 단일 컬럼으로 처리

    words = page.extract_words()
    if len(words) < 10:
        return False

    left_bound  = page.width * 0.40   # 왼쪽 40% 이내
    right_bound = page.width * 0.60   # 오른쪽 60% 이후

    far_left  = sum(1 for w in words if w["x0"] < left_bound)
    far_right = sum(1 for w in words if w["x0"] > right_bound)

    return (far_left  >= len(words) * 0.25 and
            far_right >= len(words) * 0.25)


def _extract_page_text(page) -> str:
    """
    2-컬럼 레이아웃이면 왼쪽 → 오른쪽 순으로 이어붙여 반환.
    단일 컬럼이면 extract_text() 결과를 그대로 반환.
    """
    if _is_two_column_page(page):
        mid_x  = page.width / 2
        left_text  = page.crop((0,     0, mid_x,       page.height)).extract_text() or ""
        right_text = page.crop((mid_x, 0, page.width,  page.height)).extract_text() or ""
        return left_text.strip() + "\n" + right_text.strip()

    return page.extract_text() or ""


class DocumentManager:
    """PDF 업로드, 파싱, 청킹, 임베딩 및 DB 저장을 총괄하는 서비스"""
    
    def __init__(self, doc_repo: IDocumentRepository, embedder: IEmbedder):
        self.doc_repo = doc_repo
        self.embedder = embedder

    def process_document(
        self, 
        tenant_id: str, 
        file_name: str, 
        file_content: bytes, 
        chunk_size: int = 500, 
        chunk_overlap: int = 50
    ) -> str:
        """문서 처리 파이프라인 실행"""
        # 1. 문서 해시 계산 및 저장 (중복 방지)
        import hashlib
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        doc = self.doc_repo.get_document_by_hash(tenant_id, file_hash)
        if not doc:
            doc = Document(
                document_id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                file_name=file_name,
                file_hash=file_hash,
                uploaded_at=datetime.datetime.now()
            )
            self.doc_repo.save_document(doc)
        
        # 2. 문서 버전 생성 (동일 설정 버전이 있으면 재사용)
        existing_version = self.doc_repo.get_version_by_config(
            doc.document_id, self.embedder.get_model_name(), chunk_size, chunk_overlap
        )
        if existing_version:
            return existing_version.doc_version_id  # 이미 처리된 버전 → 스킵

        doc_version = DocumentVersion(
            doc_version_id=str(uuid.uuid4()),
            document_id=doc.document_id,
            parser_name="pdfplumber",
            parser_version="0.12.1",
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            embedding_model=self.embedder.get_model_name(),
            created_at=datetime.datetime.now()
        )
        self.doc_repo.save_document_version(doc_version)

        # 3. PDF 파싱 (텍스트 추출)
        page_texts = []  # List[(page_num, text)]

        from io import BytesIO
        with pdfplumber.open(BytesIO(file_content)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = _extract_page_text(page)
                page_texts.append((i + 1, text))

        # 4. 청킹 (LangChain Splitter 활용)
        # 전체 텍스트를 이어 붙인 뒤 청킹 → 페이지 경계에서 문맥 단절 방지
        # 각 페이지 텍스트에 페이지 마커를 삽입하여 나중에 청크별 페이지를 역추적
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # 페이지 마커와 함께 전체 텍스트 구성
        PAGE_MARKER = "\n<<<PAGE:{page}>>>\n"
        marked_pages = []
        for page_num, text in page_texts:
            if text.strip():
                marked_pages.append(PAGE_MARKER.format(page=page_num) + text)
        full_marked_text = "\n".join(marked_pages)

        raw_chunks = splitter.split_text(full_marked_text)

        chunks_to_save = []
        chunk_idx = 0

        for raw_chunk in raw_chunks:
            # 페이지 마커 제거 및 청크 내 등장한 페이지 번호 추출
            page_nums = [int(m) for m in re.findall(r'<<<PAGE:(\d+)>>>', raw_chunk)]
            clean_chunk = re.sub(r'<<<PAGE:\d+>>>\n?', '', raw_chunk).strip()

            if not clean_chunk:
                continue

            # 해당 청크의 대표 페이지: 마커가 있으면 첫 번째, 없으면 이전 페이지 유지
            representative_page = page_nums[0] if page_nums else (page_texts[0][0] if page_texts else 1)

            # 임베딩 생성
            vector = self.embedder.embed_text(clean_chunk)

            chunk_obj = Chunk(
                chunk_id=str(uuid.uuid4()),
                doc_version_id=doc_version.doc_version_id,
                page_number=representative_page,
                chunk_index=chunk_idx,
                text=clean_chunk,
                token_count=len(clean_chunk.split()),  # 단어 수로 근사
                embedding_vector=vector,
                created_at=datetime.datetime.now()
            )
            chunks_to_save.append(chunk_obj)
            chunk_idx += 1
        
        # 5. DB 저장
        self.doc_repo.save_chunks(chunks_to_save)
        
        return doc_version.doc_version_id
