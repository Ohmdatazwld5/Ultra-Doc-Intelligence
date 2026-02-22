"""
RAG (Retrieval-Augmented Generation) Engine for Ultra Doc-Intelligence.
Handles embedding generation, vector storage, retrieval, and answer generation.
Enhanced with hybrid search (BM25 + semantic) and cross-encoder reranking.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import json
import hashlib

from openai import OpenAI
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from app.document_processor import TextChunk, ParsedDocument

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Represents a retrieved chunk with similarity score."""
    chunk: TextChunk
    similarity_score: float
    rank: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.chunk.content,
            "chunk_id": self.chunk.chunk_id,
            "similarity_score": round(self.similarity_score, 4),
            "rank": self.rank,
            "page_number": self.chunk.page_number
        }


@dataclass
class AnswerResult:
    """Represents the result of a question-answering query."""
    question: str
    answer: str
    confidence_score: float
    retrieved_chunks: List[RetrievedChunk]
    source_text: str
    is_answerable: bool
    guardrail_triggered: bool
    guardrail_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "confidence_score": round(self.confidence_score, 4),
            "is_answerable": self.is_answerable,
            "guardrail_triggered": self.guardrail_triggered,
            "guardrail_reason": self.guardrail_reason,
            "source_text": self.source_text,
            "sources": [rc.to_dict() for rc in self.retrieved_chunks],
            "metadata": self.metadata
        }


class EmbeddingEngine:
    """
    Handles embedding generation using local sentence-transformers.
    Supports BGE models with query prefix for better retrieval.
    Includes caching for efficiency.
    """
    
    # BGE models require a query prefix for optimal performance
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    
    def __init__(self, model: str = "BAAI/bge-base-en-v1.5"):
        self.model_name = model
        self.model = SentenceTransformer(model)
        self._embedding_cache: Dict[str, List[float]] = {}
        # Check if this is a BGE model
        self._is_bge = "bge" in model.lower()
        logger.info(f"Loaded embedding model: {model} (BGE mode: {self._is_bge})")
    
    def embed_text(self, text: str, is_query: bool = False) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            is_query: If True and using BGE model, adds query prefix
        """
        # Add BGE query prefix if needed
        embed_text = text
        if is_query and self._is_bge:
            embed_text = self.BGE_QUERY_PREFIX + text
        
        cache_key = self._get_cache_key(embed_text)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        embedding = self.model.encode(embed_text, convert_to_numpy=True).tolist()
        self._embedding_cache[cache_key] = embedding
        return embedding
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a query (uses prefix for BGE models)."""
        return self.embed_text(query, is_query=True)
    
    def embed_texts(self, texts: List[str], are_queries: bool = False) -> List[List[float]]:
        """Generate embeddings for multiple texts in batch."""
        if not texts:
            return []
        
        # Add BGE prefix if needed
        if are_queries and self._is_bge:
            texts_to_embed = [self.BGE_QUERY_PREFIX + t for t in texts]
        else:
            texts_to_embed = texts
        
        # Check cache first
        uncached_indices = []
        uncached_texts = []
        embeddings = [None] * len(texts_to_embed)
        
        for i, text in enumerate(texts_to_embed):
            cache_key = self._get_cache_key(text)
            if cache_key in self._embedding_cache:
                embeddings[i] = self._embedding_cache[cache_key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        # Batch embed uncached texts
        if uncached_texts:
            batch_embeddings = self.model.encode(uncached_texts, convert_to_numpy=True)
            for i, idx in enumerate(uncached_indices):
                embedding = batch_embeddings[i].tolist()
                embeddings[idx] = embedding
                cache_key = self._get_cache_key(texts_to_embed[idx])
                self._embedding_cache[cache_key] = embedding
        
        return embeddings
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()


class BM25Index:
    """
    BM25 keyword-based search index for hybrid retrieval.
    Complements semantic search with exact keyword matching.
    """
    
    def __init__(self):
        self._indices: Dict[str, BM25Okapi] = {}
        self._tokenized_docs: Dict[str, List[List[str]]] = {}
        self._chunk_ids: Dict[str, List[str]] = {}
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple tokenization: lowercase, alphanumeric tokens."""
        return re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    
    def build_index(self, document_id: str, chunks: List[TextChunk]) -> None:
        """Build BM25 index for a document's chunks."""
        tokenized = [self.tokenize(chunk.content) for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        
        self._tokenized_docs[document_id] = tokenized
        self._chunk_ids[document_id] = chunk_ids
        self._indices[document_id] = BM25Okapi(tokenized)
        
        logger.info(f"Built BM25 index for {document_id} with {len(chunks)} chunks")
    
    def search(self, document_id: str, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Search using BM25.
        
        Returns:
            List of (chunk_id, bm25_score) tuples
        """
        if document_id not in self._indices:
            return []
        
        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []
        
        bm25 = self._indices[document_id]
        scores = bm25.get_scores(tokenized_query)
        
        # Get top-k results
        chunk_ids = self._chunk_ids[document_id]
        scored_chunks = list(zip(chunk_ids, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        return scored_chunks[:top_k]
    
    def delete_index(self, document_id: str) -> None:
        """Remove a document's BM25 index."""
        self._indices.pop(document_id, None)
        self._tokenized_docs.pop(document_id, None)
        self._chunk_ids.pop(document_id, None)


class Reranker:
    """
    Cross-encoder reranker for improving retrieval precision.
    Takes query-chunk pairs and scores them for relevance.
    """
    
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model
        self.model = CrossEncoder(model)
        logger.info(f"Loaded reranker model: {model}")
    
    def rerank(
        self, 
        query: str, 
        chunks: List[Tuple[str, str, float]],  # (chunk_id, content, initial_score)
        top_k: int = 5
    ) -> List[Tuple[str, str, float]]:
        """
        Rerank chunks using cross-encoder.
        
        Args:
            query: User query
            chunks: List of (chunk_id, content, initial_score) tuples
            top_k: Number of top results to return
            
        Returns:
            Reranked list of (chunk_id, content, rerank_score) tuples
            Scores use sigmoid for ranking, min-max over top_k for display.
        """
        if not chunks:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [(query, content) for _, content, _ in chunks]
        
        # Score all pairs - returns raw logits
        raw_scores = np.array(self.model.predict(pairs))
        
        # Apply sigmoid for ranking (gives 0-1 absolute relevance)
        sigmoid_scores = 1 / (1 + np.exp(-raw_scores))
        
        # Sort by sigmoid scores and take top_k
        chunk_data = list(zip(
            [c[0] for c in chunks],  # chunk_ids
            [c[1] for c in chunks],  # contents
            sigmoid_scores
        ))
        chunk_data.sort(key=lambda x: x[2], reverse=True)
        top_chunks = chunk_data[:top_k]
        
        # Apply min-max scaling ONLY to top_k results for better visual spread
        # Maps scores to range [0.5, 0.95] for displayed results
        top_scores = np.array([x[2] for x in top_chunks])
        min_score = top_scores.min()
        max_score = top_scores.max()
        
        if max_score > min_score:
            # Scale to [0.5, 0.95] range - min displayed gets 50%, max gets 95%
            display_scores = 0.5 + (top_scores - min_score) / (max_score - min_score) * 0.45
        else:
            # All scores essentially equal - show them all at 90%
            display_scores = np.full_like(top_scores, 0.90)
        
        # Return with display scores
        return [
            (chunk_id, content, float(score))
            for (chunk_id, content, _), score in zip(top_chunks, display_scores)
        ]


class VectorStore:
    """
    Vector storage and retrieval using ChromaDB.
    Supports document-scoped collections for multi-document scenarios.
    """
    
    def __init__(self, persist_directory: str = "./data/vector_store"):
        self.persist_directory = persist_directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collections: Dict[str, Any] = {}
    
    def get_or_create_collection(self, document_id: str) -> Any:
        """Get or create a collection for a document."""
        collection_name = self._sanitize_collection_name(document_id)
        
        if collection_name not in self._collections:
            self._collections[collection_name] = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
        
        return self._collections[collection_name]
    
    def add_chunks(self, document_id: str, chunks: List[TextChunk], embeddings: List[List[float]]) -> None:
        """Add chunks with their embeddings to the vector store."""
        collection = self.get_or_create_collection(document_id)
        
        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number or -1,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char
            }
            for chunk in chunks
        ]
        
        # Add to collection
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(chunks)} chunks to collection {document_id}")
    
    def search(self, document_id: str, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, str, float, Dict]]:
        """
        Search for similar chunks.
        
        Returns:
            List of (chunk_id, content, similarity_score, metadata) tuples
        """
        collection = self.get_or_create_collection(document_id)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances", "metadatas"]
        )
        
        # Convert distances to similarity scores (ChromaDB returns distances for cosine)
        # For cosine distance: similarity = 1 - distance
        search_results = []
        for i in range(len(results["ids"][0])):
            chunk_id = results["ids"][0][i]
            content = results["documents"][0][i]
            distance = results["distances"][0][i]
            metadata = results["metadatas"][0][i]
            
            # Convert cosine distance to similarity
            similarity = 1 - distance
            search_results.append((chunk_id, content, similarity, metadata))
        
        return search_results
    
    def delete_collection(self, document_id: str) -> None:
        """Delete a document's collection."""
        collection_name = self._sanitize_collection_name(document_id)
        try:
            self.client.delete_collection(collection_name)
            if collection_name in self._collections:
                del self._collections[collection_name]
            logger.info(f"Deleted collection {collection_name}")
        except Exception as e:
            logger.warning(f"Failed to delete collection {collection_name}: {e}")
    
    def collection_exists(self, document_id: str) -> bool:
        """Check if a collection exists for a document."""
        collection_name = self._sanitize_collection_name(document_id)
        try:
            collections = self.client.list_collections()
            return any(c.name == collection_name for c in collections)
        except Exception:
            return False
    
    def _sanitize_collection_name(self, name: str) -> str:
        """Sanitize collection name for ChromaDB requirements."""
        # ChromaDB requires: 3-63 chars, alphanumeric + underscores/hyphens
        sanitized = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        sanitized = sanitized[:63]
        if len(sanitized) < 3:
            sanitized = sanitized + "_doc"
        return sanitized


class RAGEngine:
    """
    Main RAG engine that orchestrates retrieval and answer generation.
    Enhanced with hybrid search (BM25 + semantic) and cross-encoder reranking.
    Uses xAI Grok for LLM (supports reasoning and fast models).
    """
    
    # Model constants
    REASONING_MODEL = "grok-4-1-fast-reasoning"
    FAST_MODEL = "grok-4-1-fast"
    
    def __init__(
        self,
        xai_api_key: str,
        xai_base_url: str = "https://api.x.ai/v1",
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_model: str = "grok-4-1-fast-reasoning",
        llm_temperature: float = 0.1,
        vector_store_path: str = "./data/vector_store",
        retrieval_top_k: int = 5,
        retrieval_initial_k: int = 20,
        similarity_threshold: float = 0.3,
        use_hybrid_search: bool = True,
        use_reranking: bool = True,
        bm25_weight: float = 0.3
    ):
        # Use local sentence-transformers for embeddings (BGE model)
        self.embedding_engine = EmbeddingEngine(embedding_model)
        self.vector_store = VectorStore(vector_store_path)
        
        # BM25 index for hybrid search
        self.bm25_index = BM25Index()
        self.use_hybrid_search = use_hybrid_search
        self.bm25_weight = bm25_weight
        
        # Cross-encoder reranker
        self.use_reranking = use_reranking
        self.reranker = Reranker(reranker_model) if use_reranking else None
        
        # Use xAI Grok for LLM
        self.llm_client = OpenAI(api_key=xai_api_key, base_url=xai_base_url)
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.retrieval_top_k = retrieval_top_k
        self.retrieval_initial_k = retrieval_initial_k
        self.similarity_threshold = similarity_threshold
        
        # Store document metadata for retrieval
        self._document_chunks: Dict[str, Dict[str, TextChunk]] = {}
        
        logger.info(f"RAGEngine initialized: hybrid_search={use_hybrid_search}, reranking={use_reranking}")
    
    def index_document(self, document: ParsedDocument) -> Dict[str, Any]:
        """
        Index a parsed document for retrieval.
        
        Args:
            document: ParsedDocument to index
            
        Returns:
            Indexing statistics
        """
        if not document.chunks:
            raise ValueError("Document has no chunks to index")
        
        # Generate embeddings for all chunks (documents, not queries)
        chunk_texts = [chunk.content for chunk in document.chunks]
        embeddings = self.embedding_engine.embed_texts(chunk_texts, are_queries=False)
        
        # Store in vector store
        self.vector_store.add_chunks(document.document_id, document.chunks, embeddings)
        
        # Build BM25 index for hybrid search
        if self.use_hybrid_search:
            self.bm25_index.build_index(document.document_id, document.chunks)
        
        # Cache chunk data for retrieval
        self._document_chunks[document.document_id] = {
            chunk.chunk_id: chunk for chunk in document.chunks
        }
        
        return {
            "document_id": document.document_id,
            "chunks_indexed": len(document.chunks),
            "embedding_model": self.embedding_engine.model_name,
            "hybrid_search": self.use_hybrid_search,
            "reranking": self.use_reranking
        }
    
    def retrieve(self, document_id: str, query: str) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query using hybrid search and reranking.
        
        Pipeline:
        1. Semantic search (vector similarity)
        2. BM25 keyword search (if hybrid enabled)
        3. Combine scores with weighted fusion
        4. Rerank with cross-encoder (if enabled)
        
        Args:
            document_id: ID of the document to search
            query: User's question
            
        Returns:
            List of RetrievedChunk objects sorted by relevance
        """
        # Determine how many to retrieve initially (more if we're reranking)
        initial_k = self.retrieval_initial_k if self.use_reranking else self.retrieval_top_k
        
        # Step 1: Semantic search with BGE query embedding
        query_embedding = self.embedding_engine.embed_query(query)
        semantic_results = self.vector_store.search(
            document_id,
            query_embedding,
            top_k=initial_k
        )
        
        # Build a map of chunk_id -> (content, metadata, semantic_score)
        chunk_scores: Dict[str, Dict[str, Any]] = {}
        for chunk_id, content, similarity, metadata in semantic_results:
            chunk_scores[chunk_id] = {
                "content": content,
                "metadata": metadata,
                "semantic_score": similarity,
                "bm25_score": 0.0
            }
        
        # Step 2: BM25 keyword search (if hybrid enabled)
        if self.use_hybrid_search:
            bm25_results = self.bm25_index.search(document_id, query, top_k=initial_k)
            
            # Normalize BM25 scores
            if bm25_results:
                max_bm25 = max(score for _, score in bm25_results) or 1.0
                for chunk_id, bm25_score in bm25_results:
                    normalized_bm25 = bm25_score / max_bm25 if max_bm25 > 0 else 0
                    
                    if chunk_id in chunk_scores:
                        chunk_scores[chunk_id]["bm25_score"] = normalized_bm25
                    else:
                        # Chunk found by BM25 but not semantic - get content from cache
                        if document_id in self._document_chunks and chunk_id in self._document_chunks[document_id]:
                            cached_chunk = self._document_chunks[document_id][chunk_id]
                            chunk_scores[chunk_id] = {
                                "content": cached_chunk.content,
                                "metadata": {
                                    "chunk_index": cached_chunk.chunk_index,
                                    "page_number": cached_chunk.page_number or -1,
                                    "start_char": cached_chunk.start_char,
                                    "end_char": cached_chunk.end_char
                                },
                                "semantic_score": 0.0,
                                "bm25_score": normalized_bm25
                            }
        
        # Step 3: Combine scores with weighted fusion
        combined_results = []
        semantic_weight = 1.0 - self.bm25_weight
        
        for chunk_id, data in chunk_scores.items():
            combined_score = (
                semantic_weight * data["semantic_score"] +
                self.bm25_weight * data["bm25_score"]
            )
            combined_results.append((chunk_id, data["content"], combined_score, data["metadata"]))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x[2], reverse=True)
        
        # Step 4: Rerank with cross-encoder (if enabled)
        if self.use_reranking and self.reranker and combined_results:
            # Prepare for reranking
            chunks_to_rerank = [
                (chunk_id, content, score) 
                for chunk_id, content, score, _ in combined_results[:initial_k]
            ]
            
            reranked = self.reranker.rerank(query, chunks_to_rerank, top_k=self.retrieval_top_k)
            
            # Update results with reranked scores
            reranked_ids = {chunk_id for chunk_id, _, _ in reranked}
            final_results = []
            for chunk_id, content, rerank_score in reranked:
                # Find metadata
                metadata = next(
                    (data["metadata"] for cid, data in chunk_scores.items() if cid == chunk_id),
                    {}
                )
                final_results.append((chunk_id, content, rerank_score, metadata))
            combined_results = final_results
        else:
            # Just take top_k without reranking
            combined_results = combined_results[:self.retrieval_top_k]
        
        # Convert to RetrievedChunk objects
        retrieved_chunks = []
        for rank, (chunk_id, content, score, metadata) in enumerate(combined_results, 1):
            # Get full chunk data if cached
            if document_id in self._document_chunks and chunk_id in self._document_chunks[document_id]:
                chunk = self._document_chunks[document_id][chunk_id]
            else:
                # Reconstruct from search results
                chunk = TextChunk(
                    content=content,
                    chunk_id=chunk_id,
                    document_id=document_id,
                    chunk_index=metadata.get("chunk_index", rank),
                    start_char=metadata.get("start_char", 0),
                    end_char=metadata.get("end_char", len(content)),
                    page_number=metadata.get("page_number") if metadata.get("page_number", -1) != -1 else None
                )
            
            retrieved_chunks.append(RetrievedChunk(
                chunk=chunk,
                similarity_score=score,  # Now includes hybrid + rerank score
                rank=rank
            ))
        
        return retrieved_chunks
    
    def answer_question(
        self,
        document_id: str,
        question: str,
        min_confidence: float = 0.4,
        use_reasoning: bool = True
    ) -> AnswerResult:
        """
        Answer a question about a document using RAG.
        
        Args:
            document_id: ID of the document
            question: User's question
            min_confidence: Minimum confidence threshold
            use_reasoning: If True, use reasoning model (grok-4-1-fast-reasoning) for more accurate,
                          step-by-step analysis. If False, use fast model (grok-4-1-fast) for quicker responses.
            
        Returns:
            AnswerResult with answer, sources, and confidence
        """
        # Select model based on reasoning preference
        model_to_use = self.REASONING_MODEL if use_reasoning else self.FAST_MODEL
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve(document_id, question)
        
        if not retrieved_chunks:
            return AnswerResult(
                question=question,
                answer="I could not find any relevant information in the document to answer this question.",
                confidence_score=0.0,
                retrieved_chunks=[],
                source_text="",
                is_answerable=False,
                guardrail_triggered=True,
                guardrail_reason="No relevant chunks retrieved"
            )
        
        # Filter chunks below similarity threshold
        relevant_chunks = [
            rc for rc in retrieved_chunks 
            if rc.similarity_score >= self.similarity_threshold
        ]
        
        if not relevant_chunks:
            return AnswerResult(
                question=question,
                answer="The document does not appear to contain information relevant to this question.",
                confidence_score=max(rc.similarity_score for rc in retrieved_chunks),
                retrieved_chunks=retrieved_chunks[:3],  # Return top 3 anyway for transparency
                source_text="",
                is_answerable=False,
                guardrail_triggered=True,
                guardrail_reason=f"Retrieval similarity below threshold ({self.similarity_threshold})"
            )
        
        # Build context from relevant chunks
        context = self._build_context(relevant_chunks)
        
        # Generate answer using LLM (with selected model)
        answer, llm_confidence = self._generate_answer(question, context, model_to_use)
        
        # Calculate composite confidence score
        confidence_score = self._calculate_confidence(
            relevant_chunks,
            answer,
            llm_confidence
        )
        
        # Apply confidence guardrail
        if confidence_score < min_confidence:
            return AnswerResult(
                question=question,
                answer=f"I found some potentially relevant information, but my confidence is too low ({confidence_score:.1%}) to provide a reliable answer. The document may not contain clear information about this topic.",
                confidence_score=confidence_score,
                retrieved_chunks=relevant_chunks,
                source_text=context,
                is_answerable=False,
                guardrail_triggered=True,
                guardrail_reason=f"Confidence score ({confidence_score:.2f}) below threshold ({min_confidence})"
            )
        
        return AnswerResult(
            question=question,
            answer=answer,
            confidence_score=confidence_score,
            retrieved_chunks=relevant_chunks,
            source_text=context,
            is_answerable=True,
            guardrail_triggered=False,
            metadata={
                "llm_confidence": llm_confidence,
                "chunks_used": len(relevant_chunks),
                "avg_similarity": sum(rc.similarity_score for rc in relevant_chunks) / len(relevant_chunks)
            }
        )
    
    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        for rc in chunks:
            page_info = f" (Page {rc.chunk.page_number})" if rc.chunk.page_number else ""
            context_parts.append(f"[Source {rc.rank}{page_info}]\n{rc.chunk.content}")
        return "\n\n---\n\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str, model: Optional[str] = None) -> Tuple[str, float]:
        """
        Generate an answer using the LLM with strict grounding instructions.
        
        Args:
            question: User's question
            context: Retrieved context from document
            model: Model to use (defaults to self.llm_model if not specified)
        
        Returns:
            Tuple of (answer, llm_confidence)
        """
        model = model or self.llm_model
        system_prompt = """You are a precise logistics document assistant. Your role is to answer questions based ONLY on the provided document context.

CRITICAL RULES:
1. ONLY use information explicitly stated in the provided context
2. If the information is not in the context, say "Not found in document"
3. Never make assumptions or infer information not directly stated
4. Quote or paraphrase directly from the source when possible
5. If information is partially available, state what you found and what's missing

At the end of your response, on a new line, provide a confidence assessment in this exact format:
CONFIDENCE: [HIGH/MEDIUM/LOW]
- HIGH: Information is clearly and explicitly stated in the context
- MEDIUM: Information is present but requires some interpretation
- LOW: Information is partially present or unclear"""

        user_prompt = f"""Context from the document:
{context}

Question: {question}

Provide a direct, factual answer based ONLY on the above context. If the answer is not in the context, say "Not found in document"."""

        response = self.llm_client.chat.completions.create(
            model=model,
            temperature=self.llm_temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        full_response = response.choices[0].message.content.strip()
        
        # Parse confidence from response
        llm_confidence = 0.5  # Default medium
        if "CONFIDENCE:" in full_response:
            parts = full_response.rsplit("CONFIDENCE:", 1)
            answer = parts[0].strip()
            confidence_str = parts[1].strip().upper()
            if "HIGH" in confidence_str:
                llm_confidence = 0.9
            elif "MEDIUM" in confidence_str:
                llm_confidence = 0.6
            elif "LOW" in confidence_str:
                llm_confidence = 0.3
        else:
            answer = full_response
        
        return answer, llm_confidence
    
    def _calculate_confidence(
        self,
        chunks: List[RetrievedChunk],
        answer: str,
        llm_confidence: float
    ) -> float:
        """
        Calculate a composite confidence score based on multiple factors.
        
        Factors:
        1. Retrieval similarity (40% weight)
        2. Chunk agreement/overlap (20% weight)  
        3. LLM self-reported confidence (30% weight)
        4. Answer coverage heuristics (10% weight)
        """
        if not chunks:
            return 0.0
        
        # 1. Retrieval similarity score (weighted average, top chunks weighted more)
        weights = [1.0 / (i + 1) for i in range(len(chunks))]
        total_weight = sum(weights)
        retrieval_score = sum(
            rc.similarity_score * w 
            for rc, w in zip(chunks, weights)
        ) / total_weight
        
        # 2. Chunk agreement score (do chunks support each other?)
        # Simple heuristic: check for overlapping key terms
        agreement_score = self._calculate_chunk_agreement(chunks)
        
        # 3. LLM confidence
        # Already provided
        
        # 4. Answer coverage heuristics
        coverage_score = self._calculate_answer_coverage(answer, chunks)
        
        # Composite score with weights
        confidence = (
            0.40 * retrieval_score +
            0.20 * agreement_score +
            0.30 * llm_confidence +
            0.10 * coverage_score
        )
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_chunk_agreement(self, chunks: List[RetrievedChunk]) -> float:
        """Calculate agreement between chunks based on shared terms."""
        if len(chunks) < 2:
            return 0.7  # Single chunk, moderate agreement by default
        
        # Extract significant terms from each chunk
        chunk_terms = []
        for rc in chunks:
            # Simple term extraction: words of 4+ characters
            terms = set(
                word.lower() 
                for word in rc.chunk.content.split() 
                if len(word) >= 4 and word.isalnum()
            )
            chunk_terms.append(terms)
        
        # Calculate pairwise overlap
        overlaps = []
        for i in range(len(chunk_terms)):
            for j in range(i + 1, len(chunk_terms)):
                if chunk_terms[i] and chunk_terms[j]:
                    overlap = len(chunk_terms[i] & chunk_terms[j]) / min(len(chunk_terms[i]), len(chunk_terms[j]))
                    overlaps.append(overlap)
        
        if not overlaps:
            return 0.5
        
        return min(1.0, sum(overlaps) / len(overlaps) + 0.3)  # Boost base score
    
    def _calculate_answer_coverage(self, answer: str, chunks: List[RetrievedChunk]) -> float:
        """Calculate how well the answer is supported by the chunks."""
        if not answer or "not found" in answer.lower():
            return 0.3
        
        # Check if answer terms appear in chunks
        answer_terms = set(
            word.lower() 
            for word in answer.split() 
            if len(word) >= 4 and word.isalnum()
        )
        
        if not answer_terms:
            return 0.5
        
        chunk_text = " ".join(rc.chunk.content.lower() for rc in chunks)
        
        covered_terms = sum(1 for term in answer_terms if term in chunk_text)
        coverage = covered_terms / len(answer_terms)
        
        return min(1.0, coverage + 0.2)  # Slight boost
    
    def delete_document(self, document_id: str) -> None:
        """Remove a document from the index (vector store and BM25)."""
        self.vector_store.delete_collection(document_id)
        self.bm25_index.delete_index(document_id)
        if document_id in self._document_chunks:
            del self._document_chunks[document_id]
