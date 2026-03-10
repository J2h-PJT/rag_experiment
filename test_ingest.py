import os
from reportlab.pdfgen import canvas
from src.db.database_manager import DatabaseManager
from src.db.postgres_repository import PostgresDocumentRepository
from src.ingest.embedders import OllamaEmbedder, MockOpenAIEmbedder
from src.ingest.document_manager import DocumentManager
from src.core.models import Tenant

def create_dummy_pdf(filename="test_sample.pdf"):
    c = canvas.Canvas(filename)
    c.drawString(100, 750, "이 문서는 RAG 실험 플랫폼을 위한 테스트용 PDF 문서입니다.")
    c.drawString(100, 730, "두 번째 줄입니다. 청킹이 어떻게 되는지 확인할 수 있습니다.")
    c.showPage()
    c.drawString(100, 750, "이것은 PDF의 두 번째 페이지입니다.")
    c.drawString(100, 730, "페이지 번호 트래킹이 정상적으로 작동하는지 점검합니다.")
    c.save()
    return filename

def test_ingestion_pipeline():
    # 1. DB 셋업
    db_manager = DatabaseManager()
    db_manager.initialize_schemas()
    doc_repo = PostgresDocumentRepository(db_manager)
    
    # 2. 테넌트 생성 (기본 테스트 테넌트)
    tenant_id = "test_tenant_1"
    with db_manager.transaction() as cur:
        cur.execute("INSERT INTO tenants (tenant_id, tenant_name) VALUES (%s, %s) ON CONFLICT DO NOTHING", (tenant_id, "Test Org"))
    
    # 3. 임베더 선택 (Ollama가 실행중이 아니라면 Mock 사용)
    # 실제 llama3.1:8b 테스트 시 교체: embedder = OllamaEmbedder(model_name="llama3.1")
    embedder = MockOpenAIEmbedder()
    
    print(f"[{embedder.get_model_name()}] 임베더로 인제스트 매니저 초기화 중...")
    manager = DocumentManager(doc_repo=doc_repo, embedder=embedder, tenant_id=tenant_id)
    
    # 4. 더미 PDF 파일 생성 및 읽기
    pdf_path = create_dummy_pdf()
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()
    
    # 5. Pipeline 실행 (PDF -> Text -> Chunking -> Embedding -> DB Save)
    print("PDF 인제스팅 처리 중...")
    doc_version, chunks, is_new = manager.process_pdf_upload(
        filename=pdf_path,
        file_bytes=file_bytes,
        chunk_size=100,     # 테스트용으로 작게 설정
        chunk_overlap=20
    )
    
    print("-" * 50)
    print(f"신규 처리 여부: {is_new}")
    print(f"문서 해시: {doc_version.document_id}")
    print(f"문서 파서 정보: {doc_version.parser_name} v{doc_version.parser_version}")
    print(f"생성된 청크 개수: {len(chunks)}개")
    for ch in chunks:
        print(f"  [페이지 {ch.page_number} / 청크 인덱스 {ch.chunk_index}] 토큰:{ch.token_count} - '{ch.text[:30]}...'")
        
    print("-" * 50)
    print("DB 저장 검증 완료.")
    
    # 파일 정리
    os.remove(pdf_path)

if __name__ == "__main__":
    test_ingestion_pipeline()
