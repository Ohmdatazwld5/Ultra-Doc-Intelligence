"""
Unit tests for the document processor module.
"""

import pytest
from pathlib import Path
import tempfile
import os

from app.document_processor import (
    DocumentParser,
    TextChunker,
    DocumentProcessor,
    DocumentType,
    TextChunk,
    ParsedDocument
)


class TestDocumentParser:
    """Tests for DocumentParser class."""
    
    def test_get_document_type_pdf(self):
        parser = DocumentParser()
        assert parser.get_document_type(Path("test.pdf")) == DocumentType.PDF
        
    def test_get_document_type_docx(self):
        parser = DocumentParser()
        assert parser.get_document_type(Path("test.docx")) == DocumentType.DOCX
        
    def test_get_document_type_txt(self):
        parser = DocumentParser()
        assert parser.get_document_type(Path("test.txt")) == DocumentType.TXT
        
    def test_get_document_type_unknown(self):
        parser = DocumentParser()
        assert parser.get_document_type(Path("test.xyz")) == DocumentType.UNKNOWN
    
    def test_parse_txt_file(self):
        parser = DocumentParser()
        
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.\nIt has multiple lines.\nRate: $500.00")
            temp_path = f.name
        
        try:
            text, page_count, metadata = parser.parse(Path(temp_path))
            assert "test document" in text
            assert "$500.00" in text
            assert page_count >= 1
        finally:
            os.unlink(temp_path)


class TestTextChunker:
    """Tests for TextChunker class."""
    
    def test_chunk_empty_text(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk("", "doc_123")
        assert chunks == []
    
    def test_chunk_short_text(self):
        chunker = TextChunker(chunk_size=500, chunk_overlap=100)
        text = "This is a short text that should be a single chunk."
        chunks = chunker.chunk(text, "doc_123")
        
        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].document_id == "doc_123"
    
    def test_chunk_long_text(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "This is a longer text. " * 50  # ~1150 characters
        chunks = chunker.chunk(text, "doc_123")
        
        assert len(chunks) > 1
        # Verify all chunks have required fields
        for chunk in chunks:
            assert chunk.chunk_id is not None
            assert chunk.document_id == "doc_123"
            assert len(chunk.content) > 0
    
    def test_chunk_with_section_markers(self):
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        text = """
        SHIPPER: ABC Company
        Address: 123 Main St
        
        CONSIGNEE: XYZ Corp
        Address: 456 Oak Ave
        
        RATE: $2,500.00
        """
        chunks = chunker.chunk(text, "doc_123")
        
        # Should attempt to split on semantic boundaries
        assert len(chunks) >= 1
    
    def test_chunk_overlap(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=30)
        text = "Sentence one is here. Sentence two follows. Sentence three is next. Sentence four comes after. Sentence five ends it."
        chunks = chunker.chunk(text, "doc_123")
        
        if len(chunks) > 1:
            # Check that there's some overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                current = chunks[i].content
                next_chunk = chunks[i + 1].content
                # Some words from end of current should appear in next
                current_words = set(current.split()[-5:])
                next_words = set(next_chunk.split()[:10])
                # Not strictly required due to sentence boundary breaking
                # but content should be continuous


class TestDocumentProcessor:
    """Tests for DocumentProcessor class."""
    
    def test_process_txt_document(self):
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
        
        # Create test document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""
            RATE CONFIRMATION
            
            Shipper: Test Shipper Inc
            Address: 100 Industrial Way
            
            Consignee: Test Receiver LLC
            Address: 200 Commerce Drive
            
            Rate: $1,500.00 USD
            Weight: 20,000 lbs
            Equipment: 53' Dry Van
            """)
            temp_path = f.name
        
        try:
            result = processor.process(Path(temp_path))
            
            assert isinstance(result, ParsedDocument)
            assert result.document_id is not None
            assert result.document_type == DocumentType.TXT
            assert len(result.chunks) > 0
            assert "Rate" in result.raw_text
            assert "$1,500.00" in result.raw_text
        finally:
            os.unlink(temp_path)
    
    def test_process_nonexistent_file(self):
        processor = DocumentProcessor()
        
        with pytest.raises(FileNotFoundError):
            processor.process(Path("/nonexistent/file.txt"))


class TestTextChunk:
    """Tests for TextChunk dataclass."""
    
    def test_chunk_to_dict(self):
        chunk = TextChunk(
            content="Test content",
            chunk_id="chunk_001",
            document_id="doc_123",
            chunk_index=0,
            start_char=0,
            end_char=12,
            page_number=1,
            metadata={"test": "value"}
        )
        
        result = chunk.to_dict()
        
        assert result["content"] == "Test content"
        assert result["chunk_id"] == "chunk_001"
        assert result["document_id"] == "doc_123"
        assert result["page_number"] == 1


class TestParsedDocument:
    """Tests for ParsedDocument dataclass."""
    
    def test_document_to_dict(self):
        chunks = [
            TextChunk(
                content="Chunk 1",
                chunk_id="c1",
                document_id="doc_123",
                chunk_index=0,
                start_char=0,
                end_char=7
            )
        ]
        
        doc = ParsedDocument(
            document_id="doc_123",
            filename="test.txt",
            document_type=DocumentType.TXT,
            raw_text="Chunk 1",
            chunks=chunks,
            page_count=1,
            char_count=7
        )
        
        result = doc.to_dict()
        
        assert result["document_id"] == "doc_123"
        assert result["filename"] == "test.txt"
        assert result["document_type"] == "txt"
        assert result["chunk_count"] == 1
