# 🧪 Enterprise RAG Evaluation Platform

A comprehensive platform for evaluating Retrieval-Augmented Generation (RAG) systems and generating high-quality Gold Qrels datasets through semi-automated workflows.

[![GitHub](https://img.shields.io/badge/GitHub-J2h--PJT%2Frag__experiment-blue?logo=github)](https://github.com/J2h-PJT/rag_experiment)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)]()
[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)]()
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-18.1-blue?logo=postgresql)]()

---

## 🎯 Overview

This platform enables enterprises to:
- **Upload Documents**: Process PDFs with intelligent chunking and embedding
- **Generate Gold Qrels**: Create ground truth datasets through hybrid search + LLM + HITL
- **Run Experiments**: Evaluate RAG retrievers with multiple configurations
- **Compare Performance**: Visualize and analyze metrics across different runs

---

## ✨ Features

### 📄 Document Upload
- Multi-tenant support
- Configurable embedding models (bge-m3, etc.)
- Automatic PDF parsing and chunking
- Quality filtering (duplicates, noise, length)
- pgvector-based vector storage

### 🎯 Gold Qrels Builder
- Hybrid search (Vector + BM25, RRF)
- Cross-Encoder reranking
- LLM-powered relevance scoring
- Human-in-the-loop (HITL) validation
- Ground truth dataset generation

### 🚀 Experiment Runner
- Multiple retriever types (Vector, BM25, Hybrid)
- Configurable top-K and weights
- Comprehensive metrics (Recall@K, MRR@K, NDCG@K)
- LLM answer generation (optional)
- Per-question detailed analysis

### 📊 Metrics Dashboard
- Visual run comparison
- Multi-level filtering (Tenant → Experiment → Dataset)
- Bar charts for metric groups
- Copyable summary reports
- Session state management

---

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│    Streamlit UI (Frontend)          │
├─────────────────────────────────────┤
│  Upload │ Qrels │ Runner │ Dashboard│
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│  Business Logic Layer               │
├─────────────────────────────────────┤
│  Ingest  │ Qrels │ Evaluation       │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│  Repository Pattern (Data Access)   │
├─────────────────────────────────────┤
│  DocumentRepo │ QrelsRepo │ RunRepo │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│    PostgreSQL + pgvector            │
└─────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.x
- PostgreSQL 18.1 with pgvector extension
- Ollama (for LLM answer generation)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/J2h-PJT/rag_experiment.git
   cd rag_experiment
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env  # Edit with your settings
   # Required: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
   ```

5. **Initialize database**
   ```bash
   python -c "from src.db.database_manager import DatabaseManager; DatabaseManager().init_db()"
   ```

6. **Run the application**
   ```bash
   streamlit run app.py --server.port 8506
   ```

7. **Access the platform**
   ```
   http://localhost:8506
   ```

---

## 📖 Usage Guide

### 1. Document Upload
1. Select or create a tenant (organization)
2. Choose embedding model (bge-m3 recommended)
3. Upload PDF files
4. Wait for processing (automatic chunking + embedding)

### 2. Gold Qrels Builder
1. Select an experiment
2. Review hybrid search results
3. Apply reranking if needed
4. Review LLM-suggested scores
5. Validate and adjust relevance scores
6. Save the dataset

### 3. Experiment Runner
1. Select a Gold Qrels dataset
2. Configure retriever settings:
   - Type: Vector / BM25 / Hybrid
   - Top-K: 1-50
   - Weights: adjust for hybrid
   - Reranker: optional
3. Run the experiment
4. Analyze results:
   - Metrics summary
   - Per-question details
   - LLM-generated answers (if enabled)

### 4. Metrics Dashboard
1. Select Tenant → Experiment → Dataset
2. View all runs for the dataset
3. Compare metrics using bar charts
4. Click into run details
5. Copy summary for notes

---

## 📊 Evaluation Metrics

### Recall@K
- Proportion of relevant documents in top-K results
- **Target**: ≥0.8 (green), ≥0.5 (yellow), <0.5 (red)

### MRR@K (Mean Reciprocal Rank)
- Average rank of first relevant document
- **Target**: ≥0.5 (green), ≥0.25 (yellow), <0.25 (red)

### NDCG@K (Normalized Discounted Cumulative Gain)
- Ranking quality considering document positions
- **Target**: ≥0.7 (green), ≥0.4 (yellow), <0.4 (red)

---

## 🗄️ Database Schema

### Core Tables
- `tenants` - Organization management
- `documents` - Uploaded PDF metadata
- `document_versions` - Embedding model variants
- `chunks` - Text segments with vectors

### Qrels Tables
- `experiments` - Evaluation experiments
- `dataset_versions` - Gold Qrels datasets
- `gold_qrels` - Ground truth relevance pairs
- `questions` - Evaluation queries

### Run Tables
- `experiment_runs` - Experiment executions
- `experiment_configs` - Retriever configurations
- `retrieval_results` - Search rankings
- `evaluation_results` - Calculated metrics
- `generation_results` - LLM answers

---

## 🔧 Configuration

### Embedding Models
- `bge-m3`: Multilingual, 384 dimensions (recommended)
- Custom models via sentence-transformers

### Retriever Types
- **Vector**: pgvector semantic search
- **BM25**: Keyword-based ranking
- **Hybrid**: Combined with Reciprocal Rank Fusion (RRF)

### Rerankers
- **BGE v2-m3**: Multilingual, supports Korean (recommended)
- **MS-MARCO MiniLM**: English-only

### LLM
- **Ollama**: llama3.1:8b (default)
- Configurable via `src/llm/generator.py`

---

## 📈 Performance Optimization

### Database
- pgvector with HNSW indexing for vector search
- BM25 for keyword search
- Connection pooling for multi-tenant queries

### Search
- Reciprocal Rank Fusion (RRF) for hybrid search
- Configurable top-K to balance speed/quality
- Cross-Encoder reranking for ranking quality

### Caching
- Session state for UI responsiveness
- Database query optimization

---

## 🐛 Known Limitations & Future Work

### Current Limitations
- Single server deployment (no distributed setup)
- Synchronous operations (large datasets may take time)
- Manual PDF upload only (no API ingestion)

### Planned Improvements
- Time-series metric tracking
- Statistical analysis (mean, std, confidence intervals)
- Advanced filtering (by retriever type, reranker status)
- Data export (CSV, PDF)
- Real-time monitoring dashboard

---

## 📝 Project Structure

```
rag_experiment/
├── app.py                          # Streamlit main entry
├── README.md                       # This file
├── IMPLEMENTATION_REPORT.md        # Detailed completion report
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables
├── .gitignore                     # Git ignore patterns
│
└── src/
    ├── registry.py                # Dependency injection
    ├── core/                      # Data models & interfaces
    ├── db/                        # Database layer
    ├── ingest/                    # Document processing
    ├── qrels/                     # Gold Qrels generation
    ├── evaluation/                # Metrics & evaluation
    ├── llm/                       # LLM integration
    └── ui/                        # Streamlit components
```

---

## 🔐 Security Considerations

- **Database**: Use strong credentials, restrict network access
- **API Keys**: Store in `.env`, never commit
- **PDF Upload**: Validate file types, limit file size
- **Session State**: Tenant isolation in multi-tenant setup

---

## 📞 Support & Documentation

### Internal Documentation
- `IMPLEMENTATION_REPORT.md` - Complete implementation details
- `MEMORY.md` (project memory) - Project overview
- `phase4_implementation.md` - Latest features (Phase 4)

### External Resources
- [Streamlit Docs](https://docs.streamlit.io)
- [PostgreSQL Docs](https://www.postgresql.org/docs)
- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [sentence-transformers](https://www.sbert.net)

---

## 📄 License

[Add your license here]

---

## 👥 Contributors

- Claude AI Assistant (Implementation & Documentation)
- [Add your name/team]

---

## 🎉 Status

**Current Version**: 1.0.0
**Status**: ✅ Production Ready
**Last Updated**: 2026-03-10

---

## 📧 Contact

For issues, questions, or suggestions:
- GitHub Issues: [GitHub Repository](https://github.com/J2h-PJT/rag_experiment/issues)
- Email: [Add contact email]

---

**Built with ❤️ for Enterprise RAG Evaluation**
