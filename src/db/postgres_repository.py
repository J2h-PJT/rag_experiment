from typing import List, Optional, Tuple
import json
from psycopg2.extras import execute_values

from src.core.models import (
    GoldQrel, Document, DocumentVersion, Chunk, 
    Question, Tenant, DatasetVersion, ExperimentConfig
)
from src.core.interfaces import (
    IQrelsRepository, IDocumentRepository, IQuestionRepository, IExperimentRepository
)
from src.db.database_manager import DatabaseManager

class PostgresDocumentRepository(IDocumentRepository):
    """PostgreSQL 기반 엔터프라이즈 문서/청크 저장소"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    # ------------------ Tenants ------------------
    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        with self.db.transaction() as cur:
            cur.execute("SELECT * FROM tenants WHERE tenant_id = %s", (tenant_id,))
            row = cur.fetchone()
            if row:
                return Tenant(tenant_id=row['tenant_id'], tenant_name=row['tenant_name'], created_at=row['created_at'])
        return None

    def list_tenants(self) -> List[Tenant]:
        with self.db.transaction() as cur:
            cur.execute("SELECT * FROM tenants ORDER BY created_at DESC")
            return [
                Tenant(tenant_id=row['tenant_id'], tenant_name=row['tenant_name'], created_at=row['created_at'])
                for row in cur.fetchall()
            ]

    def save_tenant(self, tenant: Tenant) -> bool:
        with self.db.transaction() as cur:
            cur.execute("""
                INSERT INTO tenants (tenant_id, tenant_name, created_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (tenant_id) DO NOTHING;
            """, (tenant.tenant_id, tenant.tenant_name, tenant.created_at))
        return True

    # ------------------ Documents ------------------
    def list_documents_by_tenant(self, tenant_id: str) -> List[Document]:
        with self.db.transaction() as cur:
            cur.execute(
                "SELECT * FROM documents WHERE tenant_id = %s ORDER BY uploaded_at DESC",
                (tenant_id,)
            )
            return [
                Document(
                    document_id=str(row['document_id']), tenant_id=row['tenant_id'],
                    file_name=row['file_name'], file_hash=row['file_hash'],
                    uploaded_at=row['uploaded_at']
                )
                for row in cur.fetchall()
            ]

    def save_document(self, doc: Document) -> bool:
        """논리 문서(PDF 자체) 엔티티 저장"""
        with self.db.transaction() as cur:
            cur.execute("""
                INSERT INTO documents (document_id, tenant_id, file_name, file_hash, uploaded_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (file_hash) DO NOTHING;
            """, (
                doc.document_id, doc.tenant_id, doc.file_name, doc.file_hash, doc.uploaded_at
            ))
        return True

    def get_document_by_hash(self, tenant_id: str, file_hash: str) -> Optional[Document]:
        with self.db.transaction() as cur:
            cur.execute("SELECT * FROM documents WHERE tenant_id = %s AND file_hash = %s", (tenant_id, file_hash))
            row = cur.fetchone()
            if row:
                return Document(
                    document_id=str(row['document_id']), tenant_id=row['tenant_id'],
                    file_name=row['file_name'], file_hash=row['file_hash'],
                    uploaded_at=row['uploaded_at']
                )
        return None

    # ------------------ Document Versions ------------------
    def get_version_by_config(self, document_id: str, embedding_model: str,
                              chunk_size: int, overlap: int) -> Optional[DocumentVersion]:
        """동일 document + embedding_model + chunk_size + overlap 조합의 버전 반환 (중복 방지용)"""
        with self.db.transaction() as cur:
            cur.execute("""
                SELECT * FROM document_versions
                WHERE document_id = %s
                  AND embedding_model = %s
                  AND chunk_size = %s
                  AND overlap = %s
                ORDER BY created_at DESC LIMIT 1
            """, (document_id, embedding_model, chunk_size, overlap))
            row = cur.fetchone()
            if row:
                return DocumentVersion(
                    doc_version_id=str(row['doc_version_id']), document_id=str(row['document_id']),
                    parser_name=row['parser_name'], parser_version=row['parser_version'],
                    chunk_size=row['chunk_size'], overlap=row['overlap'],
                    embedding_model=row['embedding_model'], created_at=row['created_at']
                )
        return None

    def save_document_version(self, version: DocumentVersion) -> bool:
        """문서 버전 저장"""
        with self.db.transaction() as cur:
            cur.execute("""
                INSERT INTO document_versions (
                    doc_version_id, document_id, parser_name, parser_version,
                    chunk_size, overlap, embedding_model, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
            """, (
                version.doc_version_id, version.document_id,
                version.parser_name, version.parser_version, version.chunk_size, version.overlap,
                version.embedding_model, version.created_at
            ))
        return True

    def get_latest_version(self, document_id: str) -> Optional[DocumentVersion]:
        with self.db.transaction() as cur:
            cur.execute("""
                SELECT * FROM document_versions 
                WHERE document_id = %s 
                ORDER BY created_at DESC LIMIT 1
            """, (document_id,))
            row = cur.fetchone()
            if row:
                return DocumentVersion(
                    doc_version_id=str(row['doc_version_id']), document_id=str(row['document_id']),
                    parser_name=row['parser_name'], parser_version=row['parser_version'], 
                    chunk_size=row['chunk_size'], overlap=row['overlap'], 
                    embedding_model=row['embedding_model'], created_at=row['created_at']
                )
        return None

    # ------------------ Chunks ------------------
    def save_chunks(self, chunks: List[Chunk]) -> bool:
        """대규모 청크 대량 저장 (Bulk Insert by execute_values)"""
        if not chunks: return True
        
        insert_query = """
            INSERT INTO chunks (
                chunk_id, doc_version_id, page_number, chunk_index, 
                text, token_count, embedding_vector, created_at
            )
            VALUES %s
            ON CONFLICT (chunk_id) DO NOTHING;
        """
        
        # execute_values를 위한 데이터 튜플 리스트 준비
        values = []
        for chunk in chunks:
            # pgvector 포맷 변환 "[0.1, 0.2, ...]"
            emb_str = f"[{','.join(map(str, chunk.embedding_vector))}]" if chunk.embedding_vector else None
            
            values.append((
                chunk.chunk_id, chunk.doc_version_id, chunk.page_number, chunk.chunk_index,
                chunk.text, chunk.token_count, emb_str, chunk.created_at
            ))
            
        with self.db.transaction() as cur:
            execute_values(cur, insert_query, values, page_size=500)
            
        return True

    def get_chunks_by_version(self, doc_version_id: str) -> List[Chunk]:
        with self.db.transaction() as cur:
            cur.execute("SELECT * FROM chunks WHERE doc_version_id = %s ORDER BY chunk_index ASC", (doc_version_id,))
            chunks = []
            for row in cur.fetchall():
                emb = None
                if row['embedding_vector']:
                    emb = [float(x) for x in row['embedding_vector'].strip('[]').split(',')]
                chunks.append(Chunk(
                    chunk_id=str(row['chunk_id']), doc_version_id=str(row['doc_version_id']),
                    page_number=row['page_number'], chunk_index=row['chunk_index'],
                    text=row['text'], token_count=row['token_count'],
                    embedding_vector=emb, created_at=row['created_at']
                ))
            return chunks

    def search_chunks_by_embedding(self, tenant_id: str, embedding: List[float], embedding_model: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """pgvector `<=>` 연산자를 이용한 Cosine Distance 검색 (동일 embedding_model 필터링)"""
        emb_str = f"[{','.join(map(str, embedding))}]"

        # 문서당 최신 doc_version만 검색 (중복 업로드된 버전 제외)
        query = """
            SELECT c.*, (c.embedding_vector <=> %s::vector) AS distance
            FROM chunks c
            JOIN document_versions dv ON c.doc_version_id = dv.doc_version_id
            JOIN documents d ON dv.document_id = d.document_id
            WHERE d.tenant_id = %s
              AND dv.embedding_model = %s
              AND dv.doc_version_id = (
                  SELECT doc_version_id FROM document_versions
                  WHERE document_id = d.document_id AND embedding_model = %s
                  ORDER BY created_at DESC LIMIT 1
              )
            ORDER BY distance ASC
            LIMIT %s;
        """
        results = []
        with self.db.transaction() as cur:
            cur.execute(query, (emb_str, tenant_id, embedding_model, embedding_model, top_k))
            for row in cur.fetchall():
                distances = float(row['distance'])
                emb = [float(x) for x in row['embedding_vector'].strip('[]').split(',')] if row['embedding_vector'] else None
                chunk = Chunk(
                    chunk_id=str(row['chunk_id']), doc_version_id=str(row['doc_version_id']),
                    page_number=row['page_number'], chunk_index=row['chunk_index'],
                    text=row['text'], token_count=row['token_count'],
                    embedding_vector=emb, created_at=row['created_at']
                )
                results.append((chunk, distances))
        return results

    def list_all_chunks_by_tenant(self, tenant_id: str) -> List[Chunk]:
        """테넌트 소유 청크 조회 (BM25 전용) — 문서당 최신 doc_version만 사용"""
        query = """
            SELECT c.* FROM chunks c
            JOIN document_versions dv ON c.doc_version_id = dv.doc_version_id
            JOIN documents d ON dv.document_id = d.document_id
            WHERE d.tenant_id = %s
              AND dv.doc_version_id = (
                  SELECT doc_version_id FROM document_versions dv2
                  WHERE dv2.document_id = d.document_id
                  ORDER BY dv2.created_at DESC LIMIT 1
              )
            ORDER BY c.created_at ASC;
        """
        chunks = []
        with self.db.transaction() as cur:
            cur.execute(query, (tenant_id,))
            for row in cur.fetchall():
                emb = None
                if row['embedding_vector']:
                    emb = [float(x) for x in row['embedding_vector'].strip('[]').split(',')]
                chunks.append(Chunk(
                    chunk_id=str(row['chunk_id']), doc_version_id=str(row['doc_version_id']),
                    page_number=row['page_number'], chunk_index=row['chunk_index'],
                    text=row['text'], token_count=row['token_count'],
                    embedding_vector=emb, created_at=row['created_at']
                ))
        return chunks

class PostgresQrelsRepository(IQrelsRepository):
    """PostgreSQL 기반 골드 정답 저장소"""
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def save_dataset_version(self, version: DatasetVersion) -> bool:
        with self.db.transaction() as cur:
            cur.execute("""
                INSERT INTO dataset_versions (dataset_id, experiment_id, name, version, chunk_config, embedding_model, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (dataset_id) DO UPDATE SET name = EXCLUDED.name, version = EXCLUDED.version;
            """, (
                version.dataset_id, version.experiment_id, version.name, version.version,
                json.dumps(version.chunk_config), version.embedding_model, version.created_at
            ))
        return True

    def get_dataset_version(self, dataset_id: str) -> Optional[DatasetVersion]:
        with self.db.transaction() as cur:
            cur.execute("SELECT * FROM dataset_versions WHERE dataset_id = %s", (dataset_id,))
            row = cur.fetchone()
            if row:
                return DatasetVersion(
                    dataset_id=str(row['dataset_id']), experiment_id=str(row['experiment_id']),
                    name=row['name'], version=row['version'],
                    chunk_config=row['chunk_config'], embedding_model=row['embedding_model'],
                    created_at=row['created_at']
                )
        return None

    def save_qrel(self, record: GoldQrel) -> bool:
        with self.db.transaction() as cur:
            cur.execute("""
                INSERT INTO gold_qrels (dataset_id, question_id, chunk_id, llm_score, operator_score)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (dataset_id, question_id, chunk_id) DO UPDATE
                    SET llm_score = EXCLUDED.llm_score,
                        operator_score = EXCLUDED.operator_score;
            """, (
                record.dataset_id, record.question_id, record.chunk_id,
                int(record.llm_score or 0), int(record.operator_score or 0)
            ))
        return True

    def get_qrels_by_dataset(self, dataset_id: str) -> List[GoldQrel]:
        with self.db.transaction() as cur:
            cur.execute("SELECT * FROM gold_qrels WHERE dataset_id = %s", (dataset_id,))
            return [
                GoldQrel(
                    dataset_id=str(row['dataset_id']), question_id=str(row['question_id']),
                    chunk_id=str(row['chunk_id']),
                    llm_score=row['llm_score'], operator_score=row['operator_score']
                ) for row in cur.fetchall()
            ]

    def get_qrel_for_question(self, dataset_id: str, question_id: str) -> List[GoldQrel]:
        with self.db.transaction() as cur:
            cur.execute("SELECT * FROM gold_qrels WHERE dataset_id = %s AND question_id = %s", (dataset_id, question_id))
            return [
                GoldQrel(
                    dataset_id=str(row['dataset_id']), question_id=str(row['question_id']),
                    chunk_id=str(row['chunk_id']),
                    llm_score=row['llm_score'], operator_score=row['operator_score']
                ) for row in cur.fetchall()
            ]

    def list_datasets_by_experiment(self, experiment_id: str) -> List[DatasetVersion]:
        """실험에 속한 데이터셋 버전 목록 조회"""
        with self.db.transaction() as cur:
            cur.execute("""
                SELECT dv.*, COUNT(gq.chunk_id) AS qrel_count
                FROM dataset_versions dv
                LEFT JOIN gold_qrels gq ON dv.dataset_id = gq.dataset_id
                WHERE dv.experiment_id = %s
                GROUP BY dv.dataset_id
                ORDER BY dv.created_at DESC
            """, (experiment_id,))
            return [
                DatasetVersion(
                    dataset_id=str(row['dataset_id']), experiment_id=str(row['experiment_id']),
                    name=row['name'], version=row['version'],
                    chunk_config=row['chunk_config'] or {}, embedding_model=row['embedding_model'],
                    created_at=row['created_at']
                ) for row in cur.fetchall()
            ]

class PostgresQuestionRepository(IQuestionRepository):
    """PostgreSQL 기반 질문 저장소"""
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def save_question(self, question: Question) -> bool:
        with self.db.transaction() as cur:
            cur.execute("""
                INSERT INTO questions (question_id, experiment_id, query_text, created_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (question_id) DO UPDATE SET query_text = EXCLUDED.query_text;
            """, (
                question.question_id, question.experiment_id, question.query_text, question.created_at
            ))
        return True

    def get_question(self, tenant_id: str, question_id: str) -> Optional[Question]:
        with self.db.transaction() as cur:
            cur.execute("SELECT * FROM questions WHERE question_id = %s", (question_id,))
            row = cur.fetchone()
            if row:
                return Question(
                    question_id=str(row['question_id']), experiment_id=str(row['experiment_id']),
                    query_text=row['query_text'], created_at=row['created_at']
                )
        return None

    def list_questions(self, tenant_id: str) -> List[Question]:
        with self.db.transaction() as cur:
            cur.execute("""
                SELECT q.* FROM questions q
                JOIN experiments e ON q.experiment_id = e.experiment_id
                WHERE e.tenant_id = %s
                ORDER BY q.created_at DESC
            """, (tenant_id,))
            return [
                Question(
                    question_id=str(row['question_id']), experiment_id=str(row['experiment_id']),
                    query_text=row['query_text'], created_at=row['created_at']
                ) for row in cur.fetchall()
            ]

class PostgresExperimentRepository(IExperimentRepository):
    """PostgreSQL 기반 실험 저장소"""
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def save_config(self, config: ExperimentConfig) -> bool:
        with self.db.transaction() as cur:
            cur.execute("""
                INSERT INTO experiment_configs (
                    config_id, experiment_id, retriever_type, embedding_model,
                    chunk_size, overlap, reranker_type, llm_model, temperature, top_p,
                    fusion_weight, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (config_id) DO NOTHING;
            """, (
                config.config_id, config.experiment_id, config.retriever_type,
                config.embedding_model, config.chunk_size, config.overlap,
                config.reranker_type, config.llm_model, config.temperature,
                config.top_p, config.fusion_weight, config.created_at
            ))
        return True

    def get_config(self, tenant_id: str, experiment_id: str) -> Optional[ExperimentConfig]:
        with self.db.transaction() as cur:
            cur.execute("""
                SELECT ec.* FROM experiment_configs ec
                JOIN experiments e ON ec.experiment_id = e.experiment_id
                WHERE e.tenant_id = %s AND e.experiment_id = %s
                ORDER BY ec.created_at DESC LIMIT 1
            """, (tenant_id, experiment_id))
            row = cur.fetchone()
            if row:
                return ExperimentConfig(
                    config_id=str(row['config_id']), experiment_id=str(row['experiment_id']),
                    retriever_type=row['retriever_type'] or 'hybrid',
                    embedding_model=row['embedding_model'] or 'bge-m3',
                    chunk_size=row['chunk_size'] or 500, overlap=row['overlap'] or 50,
                    reranker_type=row['reranker_type'], llm_model=row['llm_model'] or 'llama3.1:8b',
                    temperature=row['temperature'] or 0.0, top_p=row['top_p'] or 1.0,
                    fusion_weight=row['fusion_weight'] if row['fusion_weight'] is not None else 1.0,
                    created_at=row['created_at']
                )
        return None

    def list_experiments(self, tenant_id: str) -> List[dict]:
        """테넌트별 실험 목록 조회"""
        with self.db.transaction() as cur:
            cur.execute("SELECT * FROM experiments WHERE tenant_id = %s ORDER BY created_at DESC", (tenant_id,))
            return [dict(row) for row in cur.fetchall()]

    def create_experiment(self, experiment_id: str, tenant_id: str, experiment_name: str, experiment_description: str, embedding_model: str = "bge-m3") -> bool:
        with self.db.transaction() as cur:
            cur.execute("""
                INSERT INTO experiments (experiment_id, tenant_id, name, description, embedding_model)
                VALUES (%s, %s, %s, %s, %s)
            """, (experiment_id, tenant_id, experiment_name, experiment_description, embedding_model))
        return True


class PostgresRunRepository:
    """experiment_runs / retrieval_results / evaluation_results 관리"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def save_run(self, run) -> bool:
        with self.db.transaction() as cur:
            cur.execute("""
                INSERT INTO experiment_runs (run_id, config_id, dataset_id, run_name, started_at, status)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id) DO NOTHING;
            """, (run.run_id, run.config_id, run.dataset_id, run.run_name, run.started_at, run.status))
        return True

    def update_run_status(self, run_id: str, status: str, finished_at) -> bool:
        with self.db.transaction() as cur:
            cur.execute("""
                UPDATE experiment_runs SET status = %s, finished_at = %s WHERE run_id = %s
            """, (status, finished_at, run_id))
        return True

    def save_retrieval_results(self, results: list) -> bool:
        if not results:
            return True
        values = [(r.run_id, r.question_id, r.chunk_id, r.rank, r.score) for r in results]
        with self.db.transaction() as cur:
            execute_values(cur, """
                INSERT INTO retrieval_results (run_id, question_id, chunk_id, rank, score)
                VALUES %s
                ON CONFLICT (run_id, question_id, chunk_id) DO UPDATE SET rank=EXCLUDED.rank, score=EXCLUDED.score;
            """, values)
        return True

    def save_evaluation_results(self, results: list) -> bool:
        if not results:
            return True
        values = [(r.run_id, r.metric_name, r.metric_value) for r in results]
        with self.db.transaction() as cur:
            execute_values(cur, """
                INSERT INTO evaluation_results (run_id, metric_name, metric_value)
                VALUES %s
                ON CONFLICT (run_id, metric_name) DO UPDATE SET metric_value=EXCLUDED.metric_value;
            """, values)
        return True

    def save_generation_results(self, results: list) -> bool:
        if not results:
            return True
        values = [(r.run_id, r.question_id, r.generated_answer, r.latency_ms, r.token_usage) for r in results]
        with self.db.transaction() as cur:
            execute_values(cur, """
                INSERT INTO generation_results (run_id, question_id, generated_answer, latency_ms, token_usage)
                VALUES %s
                ON CONFLICT (run_id, question_id) DO UPDATE
                    SET generated_answer=EXCLUDED.generated_answer, latency_ms=EXCLUDED.latency_ms;
            """, values)
        return True

    def list_runs(self, experiment_id: str) -> List[dict]:
        with self.db.transaction() as cur:
            cur.execute("""
                SELECT er.*, ec.retriever_type, ec.embedding_model, ec.reranker_type, ec.llm_model, ec.fusion_weight
                FROM experiment_runs er
                JOIN experiment_configs ec ON er.config_id = ec.config_id
                WHERE ec.experiment_id = %s
                ORDER BY er.started_at DESC
            """, (experiment_id,))
            return [dict(row) for row in cur.fetchall()]

    def get_run_metrics(self, run_id: str) -> List[dict]:
        with self.db.transaction() as cur:
            cur.execute("""
                SELECT metric_name, metric_value FROM evaluation_results
                WHERE run_id = %s ORDER BY metric_name
            """, (run_id,))
            return [dict(row) for row in cur.fetchall()]

    def get_dataset_runs_metrics(self, dataset_id: str) -> List[dict]:
        """Dataset에 속한 모든 Run의 메트릭 조회 (비교 차트용)"""
        with self.db.transaction() as cur:
            cur.execute("""
                SELECT er.run_id, er.run_name, er.status, er.started_at,
                       ev.metric_name, ev.metric_value,
                       ec.retriever_type, ec.embedding_model, ec.reranker_type
                FROM experiment_runs er
                JOIN evaluation_results ev ON er.run_id = ev.run_id
                JOIN experiment_configs ec ON er.config_id = ec.config_id
                WHERE er.dataset_id = %s
                ORDER BY er.started_at DESC, ev.metric_name
            """, (dataset_id,))
            return [dict(row) for row in cur.fetchall()]
