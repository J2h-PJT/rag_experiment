import hashlib
from typing import List, Tuple

from src.core.models import Document, DocumentVersion, Chunk
from src.core.interfaces import IDocumentRepository, IEmbedder
from src.ingest.parsers import RealPDFExtractor, RealChunker
from src.ingest.preprocessor import TextPreprocessor
from src.ingest.quality.quality_pipeline import QualityPipeline

class DocumentManager:
    """
    RAG 인제스트 파이프라인 (PDF -> DB) 오케스트레이터.
    Enterprise 설계 원칙(문서 불변, 버전 관리, 플러거블 파서/임베더)을 준수합니다.
    """
    def __init__(self, doc_repo: IDocumentRepository, embedder: IEmbedder, tenant_id: str):
        self.doc_repo = doc_repo
        self.embedder = embedder
        self.tenant_id = tenant_id
        
        # 기본 라이브러리 연동
        self.extractor = RealPDFExtractor()
        self.preprocessor = TextPreprocessor()
        self.quality_pipeline = QualityPipeline(min_tokens=20, min_alpha_ratio=0.5)

    def _calculate_file_hash(self, file_bytes: bytes) -> str:
        return hashlib.sha256(file_bytes).hexdigest()

    def process_pdf_upload(
        self, 
        filename: str, 
        file_bytes: bytes, 
        chunk_size: int = 500, 
        chunk_overlap: int = 50
    ) -> Tuple[DocumentVersion, List[Chunk], bool]:
        """
        1. PDF 업로드 -> 8. DB 저장까지 전 과정 수행
        Return: (생성/조회된 DocumentVersion, Chunk 리스트, 새로생성여부(bool))
        """
        
        # 1. 파일 해시 계산
        file_hash = self._calculate_file_hash(file_bytes)
        file_size = len(file_bytes)
        
        # 2. 문서 자체 논리 엔티티 관리
        existing_doc = self.doc_repo.get_document_by_hash(self.tenant_id, file_hash)
        
        if not existing_doc:
            doc = Document(
                tenant_id=self.tenant_id,
                file_name=filename,
                file_hash=file_hash,
                file_size=file_size,
                file_type="pdf"
            )
            self.doc_repo.save_document(doc)
            latest_version_num = 0
        else:
            doc = existing_doc
            latest_version = self.doc_repo.get_latest_version(doc.document_id)
            latest_version_num = latest_version.version_number if latest_version else 0

            # 동일한 설정의 파서/청크/임베더 캐싱 적용 검토 (선택적 최적화)
            if latest_version and \
               latest_version.chunk_size == chunk_size and \
               latest_version.overlap == chunk_overlap and \
               latest_version.embedding_model == self.embedder.get_model_name():
                existing_chunks = self.doc_repo.get_chunks_by_version(latest_version.doc_version_id)
                # 완전히 동일한 버전을 발견하면 캐싱 스킵 반환
                return latest_version, existing_chunks, False

        # 3. 새로운 DocumentVersion 환경 정의
        new_version_num = latest_version_num + 1
        doc_version = DocumentVersion(
            document_id=doc.document_id,
            version_number=new_version_num,
            parser_name=self.extractor.parser_name,
            parser_version=self.extractor.parser_version,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            embedding_model=self.embedder.get_model_name(),
            embedding_model_ver=self.embedder.get_model_version()
        )
        self.doc_repo.save_document_version(doc_version)
        
        # 4. 청커 초기화 (매 번 청크 설정이 다를 수 있으므로 인스턴스화)
        chunker = RealChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        new_chunks = []
        global_chunk_index = 0
        
        # 5. PDF 텍스트 추출 (페이지 단위 순회)
        pages_data = self.extractor.extract_text_by_page(file_bytes)
        
        for page_data in pages_data:
            page_num = page_data["page_number"]
            page_text = page_data["text"]
            
            # 5-1. 텍스트 전처리
            page_text = self.preprocessor.preprocess(page_text)
            
            # 6. Chunking 수행
            chunk_dicts = chunker.chunk_page(page_text, page_number=page_num)
            
            # ✅ 6-1. Quality Filters 적용
            chunk_texts = [c["text"] for c in chunk_dicts]
            filtered_texts = self.quality_pipeline.filter_chunks(chunk_texts)
            
            # 7. 통과한 청크만 임베딩 + DB 저장
            for idx, filtered_text in enumerate(filtered_texts):
                chunk_hash_id = Chunk.generate_id(
                    doc_version.doc_version_id, 
                    global_chunk_index
                )
                
                try:
                    embedding_vector = self.embedder.embed_text(filtered_text)
                except Exception as e:
                    print(f"임베딩 오류 (Chunk {global_chunk_index}): {e}")
                    embedding_vector = None
                
                chunk_entity = Chunk(
                    chunk_id=chunk_hash_id,
                    doc_version_id=doc_version.doc_version_id,
                    page_number=page_num,
                    chunk_index=global_chunk_index,
                    text=filtered_text,
                    token_count=len(filtered_text.split()),
                    embedding_vector=embedding_vector
                )
                new_chunks.append(chunk_entity)
                global_chunk_index += 1
        
        # 8. 대규모 Chunk DB 저장
        if new_chunks:
            self.doc_repo.save_chunks(new_chunks)
            print(f"✅ {len(new_chunks)} 청크 저장 완료")
        else:
            print(f"⚠️ 필터링 후 저장할 청크 없음")
        
        self.quality_pipeline.reset()  # 다음 문서를 위해 초기화
        return doc_version, new_chunks, True
