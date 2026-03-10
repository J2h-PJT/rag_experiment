import uuid
import datetime
from typing import List, Optional
from src.core.models import Document, DocumentVersion, Chunk
from src.core.interfaces import IDocumentRepository, IEmbedder
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber

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
        full_text = ""
        page_texts = [] # List[(page_num, text)]
        
        from io import BytesIO
        with pdfplumber.open(BytesIO(file_content)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                page_texts.append((i + 1, text))
                full_text += text + "\n"

        # 4. 청킹 (LangChain Splitter 활용)
        # 페이지 정보를 유지하며 청킹하기 위해 단순화된 로직 적용
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        chunks_to_save = []
        chunk_idx = 0
        
        # 개별 페이지별로 청킹하여 페이지 매핑 유지
        for page_num, text in page_texts:
            page_chunks = splitter.split_text(text)
            for p_chunk in page_chunks:
                # 임베딩 생성
                vector = self.embedder.embed_text(p_chunk)
                
                chunk_obj = Chunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_version_id=doc_version.doc_version_id,
                    page_number=page_num,
                    chunk_index=chunk_idx,
                    text=p_chunk,
                    token_count=len(p_chunk), # 실제 토크나이저 사용 권장
                    embedding_vector=vector,
                    created_at=datetime.datetime.now()
                )
                chunks_to_save.append(chunk_obj)
                chunk_idx += 1
        
        # 5. DB 저장
        self.doc_repo.save_chunks(chunks_to_save)
        
        return doc_version.doc_version_id
