"""
Pydantic models for API request/response schemas.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"


# ============== Upload Endpoint Models ==============

class UploadResponse(BaseModel):
    """Response model for document upload."""
    success: bool
    document_id: str
    filename: str
    message: str
    stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Document processing statistics"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "document_id": "rate_confirmation_abc123",
                "filename": "rate_confirmation.pdf",
                "message": "Document processed successfully",
                "stats": {
                    "page_count": 2,
                    "chunk_count": 8,
                    "char_count": 4500
                }
            }
        }


# ============== Ask Endpoint Models ==============

class AskRequest(BaseModel):
    """Request model for question answering."""
    document_id: str = Field(..., description="ID of the uploaded document")
    question: str = Field(..., description="Natural language question about the document")
    min_confidence: Optional[float] = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold (0-1)"
    )
    use_reasoning: Optional[bool] = Field(
        default=True,
        description="Use reasoning model (slower but more accurate) or fast model (quicker responses)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "rate_confirmation_abc123",
                "question": "What is the carrier rate?",
                "min_confidence": 0.4,
                "use_reasoning": True
            }
        }


class SourceChunk(BaseModel):
    """Source chunk information."""
    content: str
    chunk_id: str
    similarity_score: float
    rank: int
    page_number: Optional[int] = None


class AskResponse(BaseModel):
    """Response model for question answering."""
    success: bool
    question: str
    answer: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    is_answerable: bool
    guardrail_triggered: bool
    guardrail_reason: Optional[str] = None
    source_text: str
    sources: List[SourceChunk]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "question": "What is the carrier rate?",
                "answer": "The carrier rate is $2,500.00 for this shipment.",
                "confidence_score": 0.87,
                "is_answerable": True,
                "guardrail_triggered": False,
                "guardrail_reason": None,
                "source_text": "[Source 1] Rate: $2,500.00 USD...",
                "sources": [
                    {
                        "content": "Rate: $2,500.00 USD. Total: $2,500.00",
                        "chunk_id": "abc123",
                        "similarity_score": 0.92,
                        "rank": 1,
                        "page_number": 1
                    }
                ],
                "metadata": {
                    "llm_confidence": 0.9,
                    "chunks_used": 3
                }
            }
        }


# ============== Extract Endpoint Models ==============

class ExtractRequest(BaseModel):
    """Request model for structured extraction."""
    document_id: str = Field(..., description="ID of the uploaded document")
    use_llm: Optional[bool] = Field(
        default=True,
        description="Whether to use LLM for enhanced extraction"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "rate_confirmation_abc123",
                "use_llm": True
            }
        }


class ShipmentDataResponse(BaseModel):
    """Structured shipment data response."""
    shipment_id: Optional[str] = None
    shipper: Optional[str] = None
    consignee: Optional[str] = None
    pickup_datetime: Optional[str] = None
    delivery_datetime: Optional[str] = None
    equipment_type: Optional[str] = None
    mode: Optional[str] = None
    rate: Optional[str] = None
    currency: Optional[str] = None
    weight: Optional[str] = None
    carrier_name: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "shipment_id": "PRO-123456",
                "shipper": "ABC Manufacturing, 123 Industrial Way, Chicago, IL",
                "consignee": "XYZ Distribution, 456 Commerce Dr, Dallas, TX",
                "pickup_datetime": "2024-03-15 08:00",
                "delivery_datetime": "2024-03-17 14:00",
                "equipment_type": "53' Dry Van",
                "mode": "TL",
                "rate": "2500.00",
                "currency": "USD",
                "weight": "42000 lbs",
                "carrier_name": "FastFreight Logistics"
            }
        }


class ExtractionMetadata(BaseModel):
    """Metadata about the extraction process."""
    extraction_confidence: float
    fields_extracted: int
    fields_total: int
    extraction_notes: List[str]


class ExtractResponse(BaseModel):
    """Response model for structured extraction."""
    success: bool
    document_id: str
    data: ShipmentDataResponse
    metadata: ExtractionMetadata
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "document_id": "rate_confirmation_abc123",
                "data": {
                    "shipment_id": "PRO-123456",
                    "shipper": "ABC Manufacturing",
                    "consignee": "XYZ Distribution",
                    "pickup_datetime": "2024-03-15 08:00",
                    "delivery_datetime": "2024-03-17 14:00",
                    "equipment_type": "Dry Van",
                    "mode": "TL",
                    "rate": "2500.00",
                    "currency": "USD",
                    "weight": "42000 lbs",
                    "carrier_name": "FastFreight Logistics"
                },
                "metadata": {
                    "extraction_confidence": 0.85,
                    "fields_extracted": 11,
                    "fields_total": 11,
                    "extraction_notes": ["LLM extraction completed"]
                }
            }
        }


# ============== Error Models ==============

class ErrorResponse(BaseModel):
    """Standard error response."""
    success: bool = False
    error: str
    detail: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "Document not found",
                "detail": "No document with ID 'xyz' exists"
            }
        }


# ============== Health Check Models ==============

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    components: Dict[str, str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "components": {
                    "api": "ok",
                    "vector_store": "ok",
                    "llm": "ok"
                }
            }
        }


# ============== Document Info Models ==============

class DocumentInfo(BaseModel):
    """Information about an uploaded document."""
    document_id: str
    filename: str
    document_type: str
    page_count: int
    chunk_count: int
    char_count: int
    uploaded_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Response with list of documents."""
    success: bool
    documents: List[DocumentInfo]
    total: int
