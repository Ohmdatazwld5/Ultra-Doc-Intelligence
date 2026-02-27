# Ultra Doc-Intelligence

> AI-powered logistics document intelligence system with RAG, GraphRAG, guardrails, and structured extraction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red.svg)](https://streamlit.io/)
[![Node.js](https://img.shields.io/badge/Node.js-18+-339933.svg)](https://nodejs.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)

## 🎯 Overview

Ultra Doc-Intelligence is a POC AI system that enables users to upload logistics documents (Rate Confirmations, BOLs, Shipment Instructions, Invoices) and interact with them using natural language. The system combines traditional RAG for factual Q&A with GraphRAG for relationship-aware queries, plus confidence scoring and hallucination guardrails.

### Key Features

- **📄 Document Processing**: Parse PDF, DOCX, and TXT files with intelligent chunking
- **💬 Natural Language Q&A**: Ask questions and get grounded answers from document context
- **�️ GraphRAG**: Knowledge graph-based retrieval for relationship queries across entities
- **�🛡️ Hallucination Guardrails**: Multiple validation layers to ensure answer quality
- **📊 Confidence Scoring**: Composite scoring based on retrieval, grounding, and LLM confidence
- **📋 Structured Extraction**: Extract standardized shipment data to JSON format
- **🖥️ Minimal UI**: Streamlit-based interface for document interaction

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Streamlit UI                                │
│           (Upload, Q&A, Extraction, GraphRAG, History)              │
└────────────────────────────┬────────────────────────────────────────┘
                             │ HTTP
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                                │
│                                                                     │
│  ┌────────────┐ ┌──────────┐ ┌───────────┐ ┌───────────────────┐   │
│  │POST /upload│ │POST /ask │ │POST /extract│ │POST /graph/query │   │
│  └─────┬──────┘ └────┬─────┘ └─────┬──────┘ └────────┬──────────┘   │
└────────┼─────────────┼─────────────┼─────────────────┼──────────────┘
         │             │             │                 │
         ▼             ▼             ▼                 ▼
┌────────────────┐ ┌─────────────┐ ┌───────────────┐ ┌─────────────────┐
│ Document       │ │ RAG Engine  │ │ Structured    │ │ GraphRAG Engine │
│ Processor      │ │             │ │ Extractor     │ │                 │
│                │ │ ┌─────────┐ │ │               │ │ ┌─────────────┐ │
│ - PDF Parser   │ │ │Embedding│ │ │ - Pattern     │ │ │ Knowledge   │ │
│ - DOCX Parser  │ │ │ Engine  │ │ │   Matching    │ │ │ Graph       │ │
│ - Text Chunker │ │ └─────────┘ │ │ - LLM Extract │ │ │ (NetworkX)  │ │
│                │ │ ┌─────────┐ │ │ - Field Merge │ │ └─────────────┘ │
└────────┬───────┘ │ │ Vector  │ │ │               │ │ ┌─────────────┐ │
         │         │ │ Store   │ │ └───────────────┘ │ │ Entity      │ │
         │         │ │ (Chroma)│ │                   │ │ Extraction  │ │
         │         │ └─────────┘ │                   │ │ & Linking   │ │
         │         └─────────────┘                   │ └─────────────┘ │
         │                                           └─────────────────┘
         │           │ │ Store     │ │
         │           │ │ (Chroma)  │ │
         │           │ └───────────┘ │
         │           │ ┌───────────┐ │
         │           │ │ Answer    │ │
         │           │ │ Generator │ │
         │           │ └─────┬─────┘ │
         │           └───────┼───────┘
         │                   │
         │                   ▼
         │           ┌───────────────┐
         │           │  Guardrails   │
         │           │               │
         │           │ - Retrieval   │
         │           │   Threshold   │
         │           │ - Confidence  │
         │           │   Threshold   │
         │           │ - Grounding   │
         │           │   Validation  │
         │           │ - Numeric     │
         │           │   Validation  │
         │           └───────────────┘
         │
         ▼
┌─────────────────────┐     ┌─────────────────────┐
│ Local Embeddings    │     │   xAI Grok LLM      │
│ (all-MiniLM-L6-v2)  │     │  (grok-4-1-fast)    │
└─────────────────────┘     └─────────────────────┘
```

---

## 📚 Technical Deep Dive

### 1. Document Processing & Chunking Strategy

**File Parsing:**
- **PDF**: Uses `pdfplumber` for layout-aware extraction with table support, falls back to `pypdf` for simpler documents
- **DOCX**: Extracts paragraphs and tables using `python-docx`
- **TXT**: Direct text reading with encoding detection

**Chunking Strategy:**
```
1. Semantic Boundary Detection
   - Split on document sections (SHIPPER, CONSIGNEE, RATE, etc.)
   - Split on page markers
   - Split on paragraph boundaries (double newlines)

2. Size-Based Refinement
   - Target chunk size: 500 characters
   - Maximum chunk: 750 characters (1.5x target)
   - Oversized chunks split on sentence boundaries

3. Overlap Application
   - 100 character overlap between chunks
   - Overlap breaks at sentence boundaries when possible
   - Preserves context continuity
```

**Why this approach:**
- Logistics documents have clear structural sections
- Semantic boundaries preserve meaningful units (a rate section stays together)
- Overlap ensures retrieval doesn't miss information at chunk boundaries
- Smaller chunks (500 chars) provide precise retrieval vs large chunks that dilute relevance

### 2. Retrieval Method (RAG)

**Embedding Generation:**
- Model: `all-MiniLM-L6-v2` (384 dimensions, runs locally, fast)
- Batch processing with caching for efficiency
- No API calls needed - fully offline embedding

**Vector Storage:**
- ChromaDB with persistent storage
- Cosine similarity metric
- Document-scoped collections (each document gets its own index)

**Retrieval Process:**
```python
1. Embed user question
2. Search top-k (default 5) chunks by cosine similarity
3. Filter chunks below similarity threshold (default 0.3)
4. Build context from remaining chunks (ranked by relevance)
5. Generate answer using filtered context
```

**Retrieval Configuration:**
| Parameter | Default | Rationale |
|-----------|---------|-----------|
| Top-K | 5 | Balance between coverage and noise |
| Similarity Threshold | 0.3 | Filters clearly irrelevant chunks |
| Chunk Size | 500 | Precise retrieval, manageable context |

### 3. Guardrails Approach

The system implements **6 guardrail layers** to prevent hallucination:

#### Guardrail 1: Retrieval Threshold
```
IF max(chunk_similarities) < 0.3:
    TRIGGER: "No relevant chunks above threshold"
    ACTION: Return "information not found" response
```

#### Guardrail 2: Confidence Threshold
```
IF composite_confidence < 0.4:
    TRIGGER: "Low confidence"
    ACTION: Warn user, provide hedged answer
```

#### Guardrail 3: Context Coverage
```
- Extract significant terms from answer
- Check what % appear in retrieved context
IF coverage < 30%:
    TRIGGER: "Poor context coverage"
```

#### Guardrail 4: Answer Grounding (N-gram Validation)
```
- Generate 2-grams and 3-grams from answer
- Check overlap with source context
IF grounding < 20%:
    TRIGGER: "Answer not well grounded"
```

#### Guardrail 5: Refusal Detection
```
- Detect when model appropriately says "not found"
- Patterns: "not found in document", "cannot find", etc.
IF refusal_detected:
    PASS: Model correctly identified missing info
```

#### Guardrail 6: Numeric Validation
```
- Extract numbers from answer (rates, weights, dates)
- Verify each number appears in source context
IF validation_rate < 80%:
    TRIGGER: "Unverified numeric values"
```

### 4. Confidence Scoring Method

**Composite Score Formula:**
```
confidence = (
    0.40 × retrieval_similarity +
    0.20 × chunk_agreement +
    0.30 × llm_confidence +
    0.10 × answer_coverage
)
```

**Component Breakdown:**

| Component | Weight | Calculation |
|-----------|--------|-------------|
| Retrieval Similarity | 40% | Weighted average of chunk similarities (top chunks weighted higher) |
| Chunk Agreement | 20% | Term overlap between retrieved chunks (do sources agree?) |
| LLM Confidence | 30% | Self-reported confidence (HIGH=0.9, MEDIUM=0.6, LOW=0.3) |
| Answer Coverage | 10% | % of answer terms found in context |

**Confidence Thresholds:**
- **High Confidence (≥75%)**: Answer directly, no warnings
- **Medium Confidence (40-75%)**: Answer with validation note
- **Low Confidence (<40%)**: Trigger guardrail, hedge answer

### 5. Structured Extraction

**Extraction Fields:**
| Field | Description | Extraction Method |
|-------|-------------|-------------------|
| shipment_id | PRO/BOL/Order number | Regex + LLM |
| shipper | Origin party/address | LLM |
| consignee | Destination party/address | LLM |
| pickup_datetime | Scheduled pickup | Regex + LLM |
| delivery_datetime | Scheduled delivery | Regex + LLM |
| equipment_type | Trailer/container type | Regex + LLM |
| mode | TL/LTL/Intermodal | Regex + LLM |
| rate | Cost/price | Regex |
| currency | USD/CAD/etc | Regex |
| weight | Total weight with unit | Regex |
| carrier_name | Carrier/trucking company | LLM |

**Extraction Pipeline:**
```
1. Pattern Extraction (Regex)
   - Extract structured fields with known patterns
   - High confidence for rate, weight, currency

2. LLM Extraction
   - JSON-mode extraction with temperature=0
   - Better for free-form fields (shipper, consignee)

3. Result Merging
   - Prefer LLM results
   - Cross-validate with regex where possible
   - Log discrepancies for review
```

### 6. GraphRAG (Knowledge Graph RAG)

**Why GraphRAG?**

Traditional RAG excels at answering questions about specific document passages but struggles with:
- **Relationship queries**: "What is the connection between the shipper and carrier?"
- **Multi-hop reasoning**: "Which locations are involved in this shipment chain?"
- **Entity-centric questions**: "Show me all information about FastFreight Logistics"

GraphRAG addresses these limitations by building a knowledge graph that captures entities and their relationships, enabling relationship-aware retrieval and reasoning.

**Architecture:**
```
Document Text
      │
      ▼
┌─────────────────────────────────────┐
│   LLM Entity & Relationship         │
│   Extraction                        │
│   - Named Entity Recognition        │
│   - Relationship Detection          │
│   - Entity Type Classification      │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│   Knowledge Graph (NetworkX)        │
│                                     │
│   Nodes: Entities with attributes   │
│   - type (SHIPPER, CARRIER, etc.)   │
│   - document_id                     │
│   - original_text                   │
│                                     │
│   Edges: Relationships              │
│   - SHIPS_TO, TRANSPORTS, USES      │
│   - HAS_RATE, LOCATED_AT, etc.      │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│   Graph Query Engine                │
│   - Entity matching from query      │
│   - Subgraph extraction             │
│   - Relationship traversal          │
│   - LLM-based answer synthesis      │
└─────────────────────────────────────┘
```

**Supported Entity Types:**
| Entity Type | Examples |
|-------------|----------|
| SHIPPER | ABC Manufacturing Co., Origin Warehouse |
| CONSIGNEE | XYZ Distribution Center, Destination Facility |
| CARRIER | FastFreight Logistics LLC, XPO Logistics |
| LOCATION | Chicago IL, Los Angeles CA, Port of Newark |
| SHIPMENT | Load #LD53657, BOL #12345 |
| EQUIPMENT | 53' Dry Van, 40' Refrigerated Container |
| RATE | $2,450.00 USD, $1.85/mile |
| DATE | 02/24/2026, February 24, 2026 |
| WEIGHT | 42,000 lbs, 19,050 kg |
| PRODUCT | Electronics, Fresh Produce, Auto Parts |

**Relationship Types:**
| Relationship | Description |
|--------------|-------------|
| SHIPS_TO | Shipper → Consignee connection |
| TRANSPORTS | Carrier → Shipment connection |
| USES | Shipment → Equipment type |
| HAS_RATE | Shipment → Rate/cost |
| LOCATED_AT | Entity → Location |
| PICKUP_AT | Shipment → Pickup location |
| DELIVERS_TO | Shipment → Delivery location |
| SCHEDULED_FOR | Shipment → Date/time |
| WEIGHS | Shipment → Weight |
| CONTAINS | Shipment → Product |

**Query Processing Pipeline:**
```
1. Query Analysis
   - Extract entities mentioned in question
   - Identify relationship keywords (e.g., "connected to", "between")

2. Graph Traversal
   - Find matching entities in knowledge graph
   - Extract relevant subgraph (1-2 hop neighbors)
   - Collect all relationships

3. Context Building
   - Format entities and relationships as structured context
   - Include entity attributes and relationship details

4. Answer Generation
   - LLM synthesizes answer from graph context
   - Provides reasoning chain showing how entities connect
```

**GraphRAG vs Traditional RAG:**

| Aspect | Traditional RAG | GraphRAG |
|--------|-----------------|----------|
| Best for | Factual questions | Relationship questions |
| Retrieval unit | Text chunks | Entities & relationships |
| Context | Sequential text | Structured graph |
| Multi-hop reasoning | Limited | Native support |
| Entity disambiguation | Weak | Strong (via entity linking) |

**Example Queries:**
- "What is the relationship between the shipper and carrier?"
- "Which locations are connected in this shipment?"
- "Show all entities related to shipment LD53657"
- "What equipment type does FastFreight Logistics use?"

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for full-stack mode)
- xAI API key (for Grok LLM)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ultra-doc-intelligence.git
cd ultra-doc-intelligence

# Create Python virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies (for full-stack mode)
cd backend && npm install && cd ..
cd frontend && npm install && cd ..

# Configure environment
cp .env.example .env
# Edit .env and add your XAI_API_KEY
```

### Running Options

#### Option 1: Full-Stack Mode (React + Node.js + FastAPI) ⭐

**Windows PowerShell:**
```powershell
.\start-fullstack.ps1
```

**Or manually in 3 terminals:**

Terminal 1 - FastAPI ML Services:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Terminal 2 - Node.js API Gateway:
```bash
cd backend && npm start
```

Terminal 3 - React Frontend:
```bash
cd frontend && npm run dev
```

**Access:**
- 🖥️ React UI: http://localhost:3000
- 🔌 Node.js Gateway: http://localhost:3001
- 📡 FastAPI Docs: http://localhost:8000/docs

#### Option 2: Streamlit Mode (Python Only)

```bash
python run.py
```
This starts both the API server and Streamlit UI simultaneously.

**Access:**
- UI: http://localhost:8501
- API Docs: http://localhost:8000/docs

**Access:**
- UI: http://localhost:8501
- API Docs: http://localhost:8000/docs

---

## 📡 API Endpoints

### POST /upload
Upload and process a logistics document.

**Request:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@rate_confirmation.pdf"
```

**Response:**
```json
{
  "success": true,
  "document_id": "rate_confirmation_abc123",
  "filename": "rate_confirmation.pdf",
  "message": "Document processed successfully",
  "stats": {
    "page_count": 2,
    "chunk_count": 8,
    "char_count": 4500
  }
}
```

### POST /ask
Ask a question about an uploaded document.

**Request:**
```json
{
  "document_id": "rate_confirmation_abc123",
  "question": "What is the carrier rate?",
  "min_confidence": 0.4
}
```

**Response:**
```json
{
  "success": true,
  "question": "What is the carrier rate?",
  "answer": "The carrier rate is $2,500.00 USD.",
  "confidence_score": 0.87,
  "is_answerable": true,
  "guardrail_triggered": false,
  "source_text": "[Source 1] Rate: $2,500.00 USD...",
  "sources": [
    {
      "content": "Rate: $2,500.00 USD. Total: $2,500.00",
      "similarity_score": 0.92,
      "rank": 1,
      "page_number": 1
    }
  ]
}
```

### POST /extract
Extract structured shipment data.

**Request:**
```json
{
  "document_id": "rate_confirmation_abc123",
  "use_llm": true
}
```

**Response:**
```json
{
  "success": true,
  "document_id": "rate_confirmation_abc123",
  "data": {
    "shipment_id": "PRO-123456",
    "shipper": "ABC Manufacturing, Chicago, IL",
    "consignee": "XYZ Distribution, Dallas, TX",
    "pickup_datetime": "2024-03-15 08:00",
    "delivery_datetime": "2024-03-17 14:00",
    "equipment_type": "53' Dry Van",
    "mode": "TL",
    "rate": "2500.00",
    "currency": "USD",
    "weight": "42000 lbs",
    "carrier_name": "FastFreight Logistics"
  },
  "metadata": {
    "extraction_confidence": 0.85,
    "fields_extracted": 11,
    "fields_total": 11
  }
}
```

---

## ⚠️ Known Failure Cases

### 1. Scanned PDFs / Images
**Issue:** OCR-quality documents have extraction errors
**Mitigation:** Consider pre-processing with OCR tools like Tesseract

### 2. Multi-column Layouts
**Issue:** Text extraction may interleave columns
**Mitigation:** pdfplumber handles most cases; complex layouts may need specialized parsing

### 3. Handwritten Content
**Issue:** Handwritten notes not extractable
**Mitigation:** Out of scope for current system

### 4. Very Long Documents (>50 pages)
**Issue:** Too many chunks, retrieval dilution
**Mitigation:** Increase top-k or implement hierarchical retrieval

### 5. Ambiguous Questions
**Issue:** "What is the date?" when multiple dates exist
**Mitigation:** Guardrails flag low confidence; user should ask specific questions

### 6. Non-English Documents
**Issue:** Embeddings optimized for English
**Mitigation:** Use multilingual embedding model if needed

---

## 🔮 Improvement Ideas

### Immediate Improvements (1-2 weeks)

| Improvement | Impact | Effort | Description |
|-------------|--------|--------|-------------|
| **Cross-encoder Reranking** | High | Low | Re-enable `ms-marco-MiniLM-L-6-v2` reranker for 10-15% precision boost. Currently disabled for faster startup. |
| **Chunk Metadata Enrichment** | Medium | Low | Add section headers, page numbers, and document structure to chunk metadata for better filtering |
| **Query Expansion** | Medium | Low | Use LLM to expand ambiguous queries ("What's the rate?" → "What is the carrier rate, total cost, or freight charge?") |
| **Streaming Responses** | Medium | Medium | Stream LLM responses for better UX on long answers |

### Short-term Improvements (1-2 months)

| Improvement | Impact | Effort | Description |
|-------------|--------|--------|-------------|
| **Redis Caching** | High | Medium | Cache embeddings and frequent Q&A pairs for 10x faster repeat queries |
| **Document Classification** | High | Medium | Auto-detect document type (BOL, Rate Con, Invoice) to apply specialized extraction templates |
| **Async Background Processing** | Medium | Medium | Process large PDFs (100+ pages) without blocking the API |
| **Multi-modal Support** | High | High | Add OCR (Tesseract/Azure) for scanned PDFs and images |
| **Batch Upload** | Medium | Low | Support uploading multiple documents at once with parallel processing |

### Medium-term Improvements (3-6 months)

| Improvement | Impact | Effort | Description |
|-------------|--------|--------|-------------|
| **Domain-tuned Embeddings** | Very High | High | Fine-tune embedding model on logistics corpus (shipping terms, carrier names, rates) for 20%+ retrieval improvement |
| **Multi-document Q&A** | High | Medium | Query across multiple related documents (e.g., "Compare rates between these 3 rate confirmations") |
| **Confidence Calibration** | Medium | Medium | Train a calibration model on user feedback to convert raw scores to true probabilities |
| **Hierarchical Retrieval** | High | Medium | Two-stage retrieval: first find relevant sections, then retrieve specific chunks |
| **Field Validation Rules** | Medium | Low | Add business rules (rate must be positive, dates must be logical, weight within range) |

### Long-term Vision (6+ months)

| Improvement | Impact | Effort | Description |
|-------------|--------|--------|-------------|
| **Knowledge Graph** | Very High | Very High | Build entity relationships across documents (carriers ↔ lanes ↔ rates ↔ shippers) for analytical queries |
| **Active Learning Loop** | High | High | Collect user corrections to continuously improve extraction accuracy |
| **Specialized Logistics LLM** | Very High | Very High | Fine-tune LLM on logistics domain for better terminology understanding |
| **Audit Trail & Compliance** | Medium | Medium | Full traceability: who uploaded, what was extracted, audit logs for regulatory compliance |
| **Real-time Document Monitoring** | High | High | Watch folders/email for new documents, auto-process and alert on anomalies |

### Architecture Evolution Ideas

```
Current State (POC):
┌─────────────┐     ┌─────────────┐
│  Streamlit  │────▶│   FastAPI   │────▶ ChromaDB + xAI Grok
└─────────────┘     └─────────────┘

Future State (Production):
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  React SPA  │────▶│ API Gateway │────▶│ Kubernetes  │
└─────────────┘     └─────────────┘     │   Cluster   │
                           │            │             │
                           │            │ ┌─────────┐ │
                           │            │ │ Worker  │ │──▶ GPU Inference
                           │            │ │  Pods   │ │
                           │            │ └─────────┘ │
                           │            │ ┌─────────┐ │
                           │            │ │  Redis  │ │──▶ Caching
                           │            │ └─────────┘ │
                           │            │ ┌─────────┐ │
                           │            │ │Pinecone │ │──▶ Vector DB
                           │            │ └─────────┘ │
                           │            └─────────────┘
                           │
                    ┌──────▼──────┐
                    │  Message    │
                    │   Queue     │──▶ Async Processing
                    │ (RabbitMQ)  │
                    └─────────────┘
```

---

## 📁 Project Structure

```
ultra-doc-intelligence/
├── app/                       # Python ML Services
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── document_processor.py  # Document parsing & chunking
│   ├── rag_engine.py          # RAG: embeddings, retrieval, generation
│   ├── graph_rag.py           # GraphRAG: knowledge graph queries
│   ├── structured_extractor.py # Structured data extraction
│   ├── guardrails.py          # Hallucination prevention
│   ├── schemas.py             # Pydantic models
│   └── main.py                # FastAPI application
├── backend/                   # Node.js API Gateway
│   ├── package.json
│   └── server.js              # Express.js gateway server
├── frontend/                  # React.js SPA
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── index.html
│   └── src/
│       ├── main.jsx           # React entry point
│       ├── App.jsx            # Main application component
│       ├── index.css          # Tailwind styles
│       └── services/
│           └── api.js         # API client service
├── data/
│   ├── uploads/               # Uploaded documents
│   └── vector_store/          # ChromaDB persistence
├── tests/
│   └── test_*.py              # Unit tests
├── streamlit_app.py           # Streamlit UI (alternative)
├── start-fullstack.ps1        # Full-stack startup script
├── requirements.txt           # Python dependencies
├── .env.example               # Environment template
└── README.md                  # This file
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

---

## � Roadmap

### Features Status

| Feature | Status | Description |
|---------|--------|-------------|
| **Node.js Backend** | ✅ Done | Express.js API gateway for scalability |
| **React.js Frontend** | ✅ Done | Modern SPA with Tailwind CSS |
| **GraphRAG** | ✅ Done | Knowledge graph retrieval |
| **Demo Mode** | ✅ Done | Fallback responses for deployment |
| **TypeScript Support** | 🔜 Planned | Full type safety across frontend and backend |
| **PostgreSQL + pgvector** | 🔜 Planned | Production-grade vector storage with SQL capabilities |
| **Authentication** | 🔜 Planned | JWT-based auth with role-based access control |
| **Multi-tenant Support** | 🔜 Planned | Organization-level document isolation |
| **Neo4j Integration** | 🔜 Planned | Persistent knowledge graphs |
| **WebSocket Updates** | 🔜 Planned | Real-time processing status |

### Architecture Evolution

**Current Stack (POC):**
```
Streamlit (UI) → FastAPI (API) → ChromaDB + NetworkX
```

**Planned Production Stack:**
```
React.js (SPA) → Node.js/Express (API Gateway) → FastAPI (ML Services)
                                              ↓
                              PostgreSQL + pgvector + Neo4j
```

**Why Node.js + React.js?**
- **Performance**: Non-blocking I/O for handling concurrent document uploads
- **Ecosystem**: Rich npm ecosystem for document handling, real-time features
- **Developer Experience**: Hot reloading, component reusability, TypeScript
- **Scalability**: Microservices-ready architecture with API gateway pattern
- **Production-Ready UI**: Replace Streamlit POC with full-featured React dashboard

---

## �📜 License

MIT License - see LICENSE file for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

Built with ❤️ for logistics intelligence
