"""
Configuration management for Ultra Doc-Intelligence system.
Handles environment variables and application settings.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # xAI Configuration
    xai_api_key: str = Field(..., description="xAI API key for Grok LLM")
    xai_base_url: str = Field(default="https://api.x.ai/v1", description="xAI API base URL")
    
    # Model Configuration
    embedding_model: str = Field(default="BAAI/bge-base-en-v1.5", description="Sentence transformer embedding model")
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", description="Cross-encoder model for reranking")
    llm_model: str = Field(default="grok-4-1-fast-reasoning", description="xAI Grok model for QA")
    llm_model_fast: str = Field(default="grok-4-1-fast", description="xAI Grok fast model")
    llm_temperature: float = Field(default=0.1, description="LLM temperature for responses")
    
    # Chunking Configuration
    chunk_size: int = Field(default=500, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=100, description="Overlap between chunks")
    
    # Retrieval Configuration
    retrieval_top_k: int = Field(default=5, description="Number of chunks to retrieve")
    retrieval_initial_k: int = Field(default=20, description="Initial retrieval count before reranking")
    similarity_threshold: float = Field(default=0.3, description="Minimum similarity for retrieval")
    use_hybrid_search: bool = Field(default=True, description="Enable BM25 + semantic hybrid search")
    use_reranking: bool = Field(default=True, description="Enable cross-encoder reranking")
    bm25_weight: float = Field(default=0.3, description="Weight for BM25 in hybrid search (0-1)")
    
    # Confidence Thresholds
    min_confidence_threshold: float = Field(default=0.4, description="Minimum confidence to answer")
    high_confidence_threshold: float = Field(default=0.75, description="High confidence threshold")
    
    # Storage Paths
    vector_store_path: str = Field(default="./data/vector_store", description="Vector store directory")
    upload_dir: str = Field(default="./data/uploads", description="Document upload directory")
    
    # Server Configuration
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        Path(self.vector_store_path).mkdir(parents=True, exist_ok=True)
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings


# Logistics-specific constants for structured extraction
SHIPMENT_FIELDS = {
    "shipment_id": {
        "description": "Unique shipment or order identifier (e.g., PRO number, BOL number, order ID)",
        "aliases": ["shipment id", "order number", "pro number", "bol number", "reference", "load number", "booking number"]
    },
    "shipper": {
        "description": "Name and/or address of the shipping party (origin)",
        "aliases": ["shipper", "ship from", "origin", "pickup location", "sender", "consignor"]
    },
    "consignee": {
        "description": "Name and/or address of the receiving party (destination)",
        "aliases": ["consignee", "ship to", "destination", "delivery location", "receiver", "deliver to"]
    },
    "pickup_datetime": {
        "description": "Scheduled pickup date and time",
        "aliases": ["pickup date", "pickup time", "ship date", "origin date", "ready date"]
    },
    "delivery_datetime": {
        "description": "Scheduled or expected delivery date and time",
        "aliases": ["delivery date", "delivery time", "due date", "arrival date", "expected delivery"]
    },
    "equipment_type": {
        "description": "Type of trailer or container (e.g., Dry Van, Reefer, Flatbed)",
        "aliases": ["equipment", "trailer type", "container type", "vehicle type", "equipment type"]
    },
    "mode": {
        "description": "Transportation mode (e.g., TL, LTL, Intermodal, Air, Ocean)",
        "aliases": ["mode", "service type", "transport mode", "shipping method", "service level"]
    },
    "rate": {
        "description": "Total rate or cost for the shipment",
        "aliases": ["rate", "total", "amount", "cost", "price", "freight charges", "line haul"]
    },
    "currency": {
        "description": "Currency of the rate (e.g., USD, CAD, EUR)",
        "aliases": ["currency", "cur", "currency code"]
    },
    "weight": {
        "description": "Total weight of the shipment with unit",
        "aliases": ["weight", "gross weight", "total weight", "lbs", "kg", "pounds"]
    },
    "carrier_name": {
        "description": "Name of the carrier or trucking company",
        "aliases": ["carrier", "carrier name", "trucking company", "transport company", "hauler"]
    }
}

# Document type patterns for intelligent processing
DOCUMENT_PATTERNS = {
    "rate_confirmation": ["rate confirmation", "rate con", "carrier confirmation", "load confirmation"],
    "bill_of_lading": ["bill of lading", "bol", "b/l", "straight bill"],
    "invoice": ["invoice", "freight bill", "billing statement"],
    "shipment_instructions": ["shipping instructions", "shipment instructions", "pickup instructions"],
    "proof_of_delivery": ["proof of delivery", "pod", "delivery receipt"]
}
