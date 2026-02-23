"""
FastAPI Backend for Ultra Doc-Intelligence.
Provides RESTful API endpoints for document upload, Q&A, and structured extraction.
"""

import os
import logging
import shutil
import threading
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from app.config import get_settings, Settings
from app.document_processor import DocumentProcessor, ParsedDocument
from app.rag_engine import RAGEngine
from app.structured_extractor import StructuredExtractor
from app.guardrails import Guardrails, AnswerValidator
from app.graph_rag import GraphRAGEngine, GraphQueryResult
from app.schemas import (
    UploadResponse, AskRequest, AskResponse, SourceChunk,
    ExtractRequest, ExtractResponse, ShipmentDataResponse, ExtractionMetadata,
    ErrorResponse, HealthResponse, DocumentInfo, DocumentListResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============== Global State ==============

class TaskStatus(str, Enum):
    """Upload task status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class UploadTask:
    """Tracks upload task progress."""
    task_id: str
    filename: str
    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0  # 0-100
    message: str = "Queued for processing"
    document_id: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class AppState:
    """Application state container."""
    def __init__(self):
        self.settings: Optional[Settings] = None
        self.document_processor: Optional[DocumentProcessor] = None
        self.rag_engine: Optional[RAGEngine] = None
        self.extractor: Optional[StructuredExtractor] = None
        self.guardrails: Optional[Guardrails] = None
        self.validator: Optional[AnswerValidator] = None
        self.graph_rag: Optional[GraphRAGEngine] = None  # GraphRAG engine
        
        # Document registry: document_id -> document metadata
        self.documents: Dict[str, Dict[str, Any]] = {}
        # Raw text cache: document_id -> raw_text
        self.document_texts: Dict[str, str] = {}
        # Upload tasks: task_id -> UploadTask
        self.upload_tasks: Dict[str, UploadTask] = {}
        self._lock = threading.Lock()

app_state = AppState()


# ============== Lifecycle Management ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Ultra Doc-Intelligence API...")
    
    try:
        settings = get_settings()
        app_state.settings = settings
        
        # Initialize document processor
        app_state.document_processor = DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        
        # Initialize RAG engine (uses xAI Grok for LLM, BGE embeddings, hybrid search + reranking)
        app_state.rag_engine = RAGEngine(
            xai_api_key=settings.xai_api_key,
            xai_base_url=settings.xai_base_url,
            embedding_model=settings.embedding_model,
            reranker_model=settings.reranker_model,
            llm_model=settings.llm_model,
            llm_temperature=settings.llm_temperature,
            vector_store_path=settings.vector_store_path,
            retrieval_top_k=settings.retrieval_top_k,
            retrieval_initial_k=settings.retrieval_initial_k,
            similarity_threshold=settings.similarity_threshold,
            use_hybrid_search=settings.use_hybrid_search,
            use_reranking=settings.use_reranking,
            bm25_weight=settings.bm25_weight
        )
        
        # Initialize structured extractor (uses xAI Grok)
        app_state.extractor = StructuredExtractor(
            api_key=settings.xai_api_key,
            llm_model=settings.llm_model,
            base_url=settings.xai_base_url
        )
        
        # Initialize GraphRAG engine (uses xAI Grok)
        app_state.graph_rag = GraphRAGEngine(
            api_key=settings.xai_api_key,
            llm_model=settings.llm_model_fast,  # Use fast model for graph queries
            base_url=settings.xai_base_url
        )
        
        # Initialize guardrails
        app_state.guardrails = Guardrails(
            retrieval_threshold=settings.similarity_threshold,
            confidence_threshold=settings.min_confidence_threshold,
            high_confidence_threshold=settings.high_confidence_threshold
        )
        app_state.validator = AnswerValidator(app_state.guardrails)
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Ultra Doc-Intelligence API...")


# ============== FastAPI App ==============

app = FastAPI(
    title="Ultra Doc-Intelligence API",
    description="""
    AI-powered logistics document intelligence system.
    
    Features:
    - Document upload and processing (PDF, DOCX, TXT)
    - Natural language Q&A with RAG
    - Structured data extraction
    - Hallucination guardrails and confidence scoring
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Dependency ==============

def get_state() -> AppState:
    """Dependency to get application state."""
    if app_state.settings is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return app_state


# ============== Health Endpoint ==============

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check system health status."""
    components = {
        "api": "ok",
        "vector_store": "ok" if app_state.rag_engine else "not_initialized",
        "llm": "ok" if app_state.extractor else "not_initialized"
    }
    
    return HealthResponse(
        status="healthy" if all(v == "ok" for v in components.values()) else "degraded",
        version="1.0.0",
        components=components
    )


# ============== Background Processing ==============

def process_document_background(task_id: str, file_path: Path, filename: str, state: AppState):
    """
    Process document in background thread.
    Updates task status as processing progresses.
    """
    task = state.upload_tasks.get(task_id)
    if not task:
        return
    
    try:
        # Update status to processing
        with state._lock:
            task.status = TaskStatus.PROCESSING
            task.progress = 10
            task.message = "Parsing document..."
        
        # Parse document
        parsed_doc: ParsedDocument = state.document_processor.process(file_path)
        
        with state._lock:
            task.progress = 40
            task.message = f"Extracted {len(parsed_doc.chunks)} chunks. Generating embeddings..."
        
        # Index in vector store (this is the slow part - embedding generation)
        index_stats = state.rag_engine.index_document(parsed_doc)
        
        with state._lock:
            task.progress = 90
            task.message = "Finalizing..."
        
        # Store document metadata
        doc_info = {
            "document_id": parsed_doc.document_id,
            "filename": filename,
            "document_type": parsed_doc.document_type.value,
            "page_count": parsed_doc.page_count,
            "chunk_count": len(parsed_doc.chunks),
            "char_count": parsed_doc.char_count,
            "file_path": str(file_path),
            "uploaded_at": datetime.now().isoformat()
        }
        
        with state._lock:
            state.documents[parsed_doc.document_id] = doc_info
            state.document_texts[parsed_doc.document_id] = parsed_doc.raw_text
            
            task.status = TaskStatus.COMPLETED
            task.progress = 100
            task.message = "Document processed successfully"
            task.document_id = parsed_doc.document_id
            task.result = {
                "success": True,
                "document_id": parsed_doc.document_id,
                "filename": filename,
                "stats": {
                    "page_count": parsed_doc.page_count,
                    "chunk_count": len(parsed_doc.chunks),
                    "char_count": parsed_doc.char_count,
                    "embedding_model": index_stats.get("embedding_model")
                }
            }
        
        logger.info(f"Background processing complete: {parsed_doc.document_id}")
        
    except Exception as e:
        logger.error(f"Background processing failed for {task_id}: {e}")
        with state._lock:
            task.status = TaskStatus.FAILED
            task.progress = 0
            task.message = f"Processing failed: {str(e)}"
            task.error = str(e)


# ============== Document Endpoints ==============

@app.post("/upload", tags=["Documents"])
async def upload_document(
    file: UploadFile = File(..., description="Logistics document (PDF, DOCX, or TXT)"),
    background_tasks: BackgroundTasks = None,
    state: AppState = Depends(get_state)
):
    """
    Upload and process a logistics document.
    
    Returns immediately with a task_id. Poll /upload/status/{task_id} for progress.
    
    The document will be:
    1. Parsed to extract text
    2. Chunked intelligently for retrieval
    3. Embedded and indexed in the vector store
    
    Supported formats: PDF, DOCX, TXT
    """
    # Validate file type
    filename = file.filename or "unknown"
    extension = Path(filename).suffix.lower()
    
    if extension not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension}. Supported: .pdf, .docx, .txt"
        )
    
    try:
        # Save uploaded file
        upload_dir = Path(state.settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{filename}"
        file_path = upload_dir / safe_filename
        
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        logger.info(f"Saved uploaded file: {file_path}")
        
        # Create task and start background processing
        task_id = str(uuid.uuid4())
        task = UploadTask(task_id=task_id, filename=filename)
        
        with state._lock:
            state.upload_tasks[task_id] = task
        
        # Start background thread for processing
        thread = threading.Thread(
            target=process_document_background,
            args=(task_id, file_path, filename, state),
            daemon=True
        )
        thread.start()
        
        logger.info(f"Started background processing: task_id={task_id}")
        
        return {
            "success": True,
            "task_id": task_id,
            "filename": filename,
            "message": "Upload received. Processing in background.",
            "status_url": f"/upload/status/{task_id}"
        }
        
    except Exception as e:
        logger.error(f"Error starting upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/upload/status/{task_id}", tags=["Documents"])
async def get_upload_status(task_id: str, state: AppState = Depends(get_state)):
    """
    Get the status of an upload task.
    
    Poll this endpoint to track processing progress.
    """
    task = state.upload_tasks.get(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    
    response = {
        "task_id": task.task_id,
        "filename": task.filename,
        "status": task.status.value,
        "progress": task.progress,
        "message": task.message
    }
    
    if task.status == TaskStatus.COMPLETED:
        response["document_id"] = task.document_id
        response["result"] = task.result
    elif task.status == TaskStatus.FAILED:
        response["error"] = task.error
    
    return response


@app.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
async def list_documents(state: AppState = Depends(get_state)):
    """List all uploaded documents."""
    documents = [
        DocumentInfo(
            document_id=doc["document_id"],
            filename=doc["filename"],
            document_type=doc["document_type"],
            page_count=doc["page_count"],
            chunk_count=doc["chunk_count"],
            char_count=doc["char_count"],
            uploaded_at=doc.get("uploaded_at")
        )
        for doc in state.documents.values()
    ]
    
    return DocumentListResponse(
        success=True,
        documents=documents,
        total=len(documents)
    )


@app.delete("/documents/{document_id}", tags=["Documents"])
async def delete_document(
    document_id: str,
    state: AppState = Depends(get_state)
):
    """Delete an uploaded document and its index."""
    if document_id not in state.documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Remove from vector store
        state.rag_engine.delete_document(document_id)
        
        # Remove file if exists
        doc_info = state.documents[document_id]
        file_path = Path(doc_info.get("file_path", ""))
        if file_path.exists():
            file_path.unlink()
        
        # Remove from registry
        del state.documents[document_id]
        if document_id in state.document_texts:
            del state.document_texts[document_id]
        
        return {"success": True, "message": f"Document {document_id} deleted"}
        
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Question Answering Endpoint ==============

@app.post("/ask", response_model=AskResponse, tags=["Q&A"])
async def ask_question(
    request: AskRequest,
    state: AppState = Depends(get_state)
):
    """
    Ask a natural language question about an uploaded document.
    
    The system will:
    1. Retrieve relevant chunks using semantic search
    2. Generate an answer grounded in the document context
    3. Apply guardrails to prevent hallucination
    4. Return the answer with confidence score and sources
    """
    # Validate document exists
    if request.document_id not in state.documents:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {request.document_id}"
        )
    
    try:
        # Get answer from RAG engine
        result = state.rag_engine.answer_question(
            document_id=request.document_id,
            question=request.question,
            min_confidence=request.min_confidence or state.settings.min_confidence_threshold,
            use_reasoning=request.use_reasoning if request.use_reasoning is not None else True
        )
        
        # Convert retrieved chunks to response format
        sources = [
            SourceChunk(
                content=rc.chunk.content[:500] + "..." if len(rc.chunk.content) > 500 else rc.chunk.content,
                chunk_id=rc.chunk.chunk_id,
                similarity_score=round(rc.similarity_score, 4),
                rank=rc.rank,
                page_number=rc.chunk.page_number
            )
            for rc in result.retrieved_chunks
        ]
        
        return AskResponse(
            success=True,
            question=result.question,
            answer=result.answer,
            confidence_score=round(result.confidence_score, 4),
            is_answerable=result.is_answerable,
            guardrail_triggered=result.guardrail_triggered,
            guardrail_reason=result.guardrail_reason,
            source_text=result.source_text[:2000] if result.source_text else "",
            sources=sources,
            metadata=result.metadata
        )
        
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Structured Extraction Endpoint ==============

@app.post("/extract", response_model=ExtractResponse, tags=["Extraction"])
async def extract_structured_data(
    request: ExtractRequest,
    state: AppState = Depends(get_state)
):
    """
    Extract structured shipment data from an uploaded document.
    
    Extracts the following fields (returns null if not found):
    - shipment_id
    - shipper
    - consignee  
    - pickup_datetime
    - delivery_datetime
    - equipment_type
    - mode
    - rate
    - currency
    - weight
    - carrier_name
    """
    # Validate document exists
    if request.document_id not in state.documents:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {request.document_id}"
        )
    
    # Get document text
    if request.document_id not in state.document_texts:
        raise HTTPException(
            status_code=500,
            detail="Document text not available"
        )
    
    try:
        document_text = state.document_texts[request.document_id]
        
        # Extract structured data
        shipment_data = state.extractor.extract(
            document_text=document_text,
            use_llm=request.use_llm if request.use_llm is not None else True
        )
        
        # Build response
        data = ShipmentDataResponse(
            shipment_id=shipment_data.shipment_id,
            shipper=shipment_data.shipper,
            consignee=shipment_data.consignee,
            pickup_datetime=shipment_data.pickup_datetime,
            delivery_datetime=shipment_data.delivery_datetime,
            equipment_type=shipment_data.equipment_type,
            mode=shipment_data.mode,
            rate=shipment_data.rate,
            currency=shipment_data.currency,
            weight=shipment_data.weight,
            carrier_name=shipment_data.carrier_name
        )
        
        metadata = ExtractionMetadata(
            extraction_confidence=round(shipment_data.extraction_confidence, 4),
            fields_extracted=shipment_data.fields_extracted,
            fields_total=shipment_data.fields_total,
            extraction_notes=shipment_data.extraction_notes
        )
        
        return ExtractResponse(
            success=True,
            document_id=request.document_id,
            data=data,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Error extracting data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== GraphRAG Endpoints ==============

class GraphIndexRequest(BaseModel):
    """Request to index a document in the knowledge graph."""
    document_id: str = Field(..., description="ID of the document to index")


class GraphIndexResponse(BaseModel):
    """Response from graph indexing."""
    success: bool
    document_id: str
    entities_count: int
    relationships_count: int
    message: str


class GraphQueryRequest(BaseModel):
    """Request to query the knowledge graph."""
    query: str = Field(..., description="Natural language query")
    max_entities: int = Field(default=20, description="Maximum entities to consider")


class GraphQueryResponse(BaseModel):
    """Response from graph query."""
    success: bool
    query: str
    answer: str
    entities_found: int
    relationships_found: int
    confidence: float
    reasoning: Optional[str] = None


class GraphStatsResponse(BaseModel):
    """Graph statistics response."""
    total_entities: int
    total_relationships: int
    documents_indexed: int
    entity_types: Dict[str, int]
    relationship_types: Dict[str, int]


@app.post("/graph/index", response_model=GraphIndexResponse, tags=["GraphRAG"])
async def graph_index_document(
    request: GraphIndexRequest,
    state: AppState = Depends(get_state)
):
    """
    Index a document into the knowledge graph.
    
    Extracts entities (shipments, shippers, carriers, locations, etc.) and 
    their relationships from the document.
    """
    if not state.graph_rag:
        raise HTTPException(status_code=503, detail="GraphRAG engine not initialized")
    
    # Check if document exists
    if request.document_id not in state.document_texts:
        raise HTTPException(status_code=404, detail=f"Document {request.document_id} not found")
    
    try:
        document_text = state.document_texts[request.document_id]
        doc_info = state.documents.get(request.document_id, {})
        document_name = doc_info.get("filename", request.document_id)
        
        # Index in knowledge graph (synchronous method)
        result = state.graph_rag.index_document(
            document_text=document_text,
            document_id=request.document_id,
            document_name=document_name
        )
        
        return GraphIndexResponse(
            success=True,
            document_id=request.document_id,
            entities_count=result.get("entities_added", result.get("entities_count", 0)),
            relationships_count=result.get("relationships_added", result.get("relationships_count", 0)),
            message=f"Indexed {result.get('entities_added', result.get('entities_count', 0))} entities and {result.get('relationships_added', result.get('relationships_count', 0))} relationships"
        )
        
    except Exception as e:
        logger.error(f"Error indexing document in graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/graph/query", response_model=GraphQueryResponse, tags=["GraphRAG"])
async def graph_query(
    request: GraphQueryRequest,
    state: AppState = Depends(get_state)
):
    """
    Query the knowledge graph using natural language.
    
    Supports relationship-based queries like:
    - "Which carriers have delivered to location X?"
    - "What shipments does shipper Y have?"
    - "Show connections between carrier A and consignee B"
    """
    if not state.graph_rag:
        raise HTTPException(status_code=503, detail="GraphRAG engine not initialized")
    
    try:
        # Synchronous query method
        result = state.graph_rag.query(question=request.query)
        
        return GraphQueryResponse(
            success=True,
            query=request.query,
            answer=result.answer,
            entities_found=len(result.entities),
            relationships_found=len(result.relationships),
            confidence=result.confidence,
            reasoning=" → ".join(result.reasoning_path) if result.reasoning_path else None
        )
        
    except Exception as e:
        logger.error(f"Error querying knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/stats", response_model=GraphStatsResponse, tags=["GraphRAG"])
async def graph_stats(state: AppState = Depends(get_state)):
    """
    Get statistics about the knowledge graph.
    """
    if not state.graph_rag:
        raise HTTPException(status_code=503, detail="GraphRAG engine not initialized")
    
    try:
        stats = state.graph_rag.get_graph_stats()
        return GraphStatsResponse(
            total_entities=stats.get("total_nodes", 0),
            total_relationships=stats.get("total_edges", 0),
            documents_indexed=stats.get("indexed_documents", 0),
            entity_types=stats.get("entity_types", {}),
            relationship_types=stats.get("relationship_types", {})
        )
        
    except Exception as e:
        logger.error(f"Error getting graph stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/visualize", tags=["GraphRAG"])
async def graph_visualize(
    max_nodes: int = 50,
    state: AppState = Depends(get_state)
):
    """
    Get graph visualization data.
    
    Returns nodes and edges suitable for visualization libraries.
    """
    if not state.graph_rag:
        raise HTTPException(status_code=503, detail="GraphRAG engine not initialized")
    
    try:
        viz_data = state.graph_rag.visualize_graph(max_nodes=max_nodes)
        return {
            "success": True,
            "nodes": viz_data["nodes"],
            "edges": viz_data["edges"],
            "node_count": len(viz_data["nodes"]),
            "edge_count": len(viz_data["edges"])
        }
        
    except Exception as e:
        logger.error(f"Error generating visualization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Error Handlers ==============

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            error=exc.detail,
            detail=str(exc)
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            success=False,
            error="Internal server error",
            detail=str(exc)
        ).model_dump()
    )


# ============== Main Entry Point ==============

def main():
    """Run the API server."""
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )


if __name__ == "__main__":
    main()
