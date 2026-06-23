"""
DUNE RAG Pipeline: Retrieval-Augmented Generation
====================================================
Autonomous document ingestion, chunking, embedding, and retrieval.
Integrates with DUNE-1900's statistical engine and DUNE-ΩΩΩ's cognitive memory.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import os
import re
import math
import threading
import time
from pathlib import Path

from dune.engine.statistical import Tokenizer, TFIDFVectorizer


# ═══════════════════════════════════════════════════════════════════════
# I. DOCUMENT PROCESSING
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Document:
    """A document with metadata for RAG."""
    content: str
    source: str
    doc_id: str = ""
    title: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunks: List['DocumentChunk'] = field(default_factory=list)

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = hashlib.sha256(
                (self.source + self.content[:100]).encode()
            ).hexdigest()[:16]

    def to_dict(self) -> Dict:
        return {
            'doc_id': self.doc_id,
            'source': self.source,
            'title': self.title,
            'timestamp': self.timestamp.isoformat(),
            'content_length': len(self.content),
            'num_chunks': len(self.chunks),
            'metadata': self.metadata,
        }


@dataclass
class DocumentChunk:
    """A chunk of a document with embedding."""
    content: str
    doc_id: str
    chunk_index: int = 0
    embedding: List[float] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}_chunk_{self.chunk_index}"


class DocumentChunker:
    """
    Splits documents into chunks with configurable strategies.
    Supports sliding window, semantic, and recursive splitting.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = Tokenizer()

    def chunk_by_tokens(self, document: Document) -> List[DocumentChunk]:
        """Split document into token-based chunks with overlap."""
        tokens = self.tokenizer(document.content)
        chunks = []
        step = self.chunk_size - self.overlap

        for i in range(0, max(len(tokens), 1), step):
            chunk_tokens = tokens[i:i + self.chunk_size]
            if not chunk_tokens:
                continue
            chunk = DocumentChunk(
                content=' '.join(chunk_tokens),
                doc_id=document.doc_id,
                chunk_index=len(chunks),
                tokens=chunk_tokens,
            )
            chunks.append(chunk)

        document.chunks = chunks
        return chunks

    def chunk_by_paragraphs(self, document: Document) -> List[DocumentChunk]:
        """Split document by paragraphs, grouping small paragraphs."""
        paragraphs = [p.strip() for p in document.content.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_tokens = self.tokenizer(para)
            if current_size + len(para_tokens) > self.chunk_size and current_chunk:
                chunk = DocumentChunk(
                    content='\n\n'.join(current_chunk),
                    doc_id=document.doc_id,
                    chunk_index=len(chunks),
                )
                chunks.append(chunk)
                current_chunk = []
                current_size = 0
            current_chunk.append(para)
            current_size += len(para_tokens)

        if current_chunk:
            chunk = DocumentChunk(
                content='\n\n'.join(current_chunk),
                doc_id=document.doc_id,
                chunk_index=len(chunks),
            )
            chunks.append(chunk)

        document.chunks = chunks
        return chunks

    def chunk_recursive(self, document: Document,
                        separators: Optional[List[str]] = None) -> List[DocumentChunk]:
        """Recursively split trying different separators."""
        if separators is None:
            separators = ['\n\n', '\n', '. ', ' ', '']

        def _split(text: str, sep_idx: int, current_chunks: List[DocumentChunk]) -> None:
            if len(self.tokenizer(text)) <= self.chunk_size or sep_idx >= len(separators):
                if text.strip():
                    current_chunks.append(DocumentChunk(
                        content=text.strip(),
                        doc_id=document.doc_id,
                        chunk_index=len(current_chunks),
                    ))
                return

            sep = separators[sep_idx]
            if not sep:
                # Character-level split as last resort
                tokens = self.tokenizer(text)
                for i in range(0, len(tokens), self.chunk_size):
                    chunk_text = ' '.join(tokens[i:i + self.chunk_size])
                    current_chunks.append(DocumentChunk(
                        content=chunk_text,
                        doc_id=document.doc_id,
                        chunk_index=len(current_chunks),
                    ))
                return

            parts = text.split(sep)
            current = []
            for part in parts:
                if not part.strip():
                    continue
                combined = current + [part] if current else [part]
                combined_text = sep.join(combined)
                if len(self.tokenizer(combined_text)) > self.chunk_size and current:
                    chunk_text = sep.join(current)
                    _split(chunk_text, sep_idx + 1, current_chunks)
                    current = [part]
                else:
                    current.append(part)

            if current:
                chunk_text = sep.join(current)
                _split(chunk_text, sep_idx + 1, current_chunks)

        chunks: List[DocumentChunk] = []
        _split(document.content, 0, chunks)
        # Re-index
        for i, c in enumerate(chunks):
            c.chunk_index = i
        document.chunks = chunks
        return chunks


# ═══════════════════════════════════════════════════════════════════════
# II. EMBEDDING ENGINE
# ═══════════════════════════════════════════════════════════════════════

class EmbeddingEngine:
    """
    Generates embeddings for text using statistical methods.
    Provides vector representations for semantic search without external models.
    Uses TF-IDF weighted vectors and SVD-like dimensionality reduction.
    """

    def __init__(self, dimensions: int = 128):
        self.dimensions = dimensions
        self.tokenizer = Tokenizer()
        self._term_vectors: Dict[str, List[float]] = {}
        self._idf_cache: Dict[str, float] = {}
        self._doc_count = 0
        self._term_doc_freq: Dict[str, int] = defaultdict(int)
        self._vocab: Set[str] = set()

    def fit(self, texts: List[str]) -> 'EmbeddingEngine':
        """Fit the embedding model on a corpus of texts."""
        for text in texts:
            tokens = set(self.tokenizer(text))
            self._doc_count += 1
            for token in tokens:
                self._term_doc_freq[token] += 1
                self._vocab.add(token)

        # Generate random but fixed projection vectors for each term
        import random
        rng = random.Random(42)  # Fixed seed for reproducibility
        for term in self._vocab:
            vec = [rng.gauss(0, 1) / math.sqrt(self.dimensions)
                   for _ in range(self.dimensions)]
            self._term_vectors[term] = vec

        self._idf_cache = {
            t: math.log((self._doc_count + 1) / (freq + 1)) + 1
            for t, freq in self._term_doc_freq.items()
        }
        return self

    def embed(self, text: str) -> List[float]:
        """Generate embedding vector for text."""
        tokens = self.tokenizer(text)
        if not tokens:
            return [0.0] * self.dimensions

        # Weighted sum of term vectors using TF-IDF weights
        vec = [0.0] * self.dimensions
        total_weight = 0.0
        term_counts = defaultdict(int)

        for t in tokens:
            term_counts[t] += 1

        for term, count in term_counts.items():
            if term not in self._term_vectors:
                continue
            tf = math.log(1 + count)
            idf = self._idf_cache.get(term, 1.0)
            weight = tf * idf
            term_vec = self._term_vectors[term]
            for i in range(self.dimensions):
                vec[i] += weight * term_vec[i]
            total_weight += weight

        # Normalize
        if total_weight > 0:
            for i in range(self.dimensions):
                vec[i] /= total_weight

        # L2 normalize
        norm = math.sqrt(sum(v ** 2 for v in vec))
        if norm > 0:
            for i in range(self.dimensions):
                vec[i] /= norm

        return vec

    def cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a ** 2 for a in vec_a))
        norm_b = math.sqrt(sum(b ** 2 for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def to_dict(self) -> Dict:
        return {
            'dimensions': self.dimensions,
            'vocab_size': len(self._vocab),
            'doc_count': self._doc_count,
        }


# ═══════════════════════════════════════════════════════════════════════
# III. VECTOR STORE
# ═══════════════════════════════════════════════════════════════════════

class VectorStore:
    """
    Simple vector store for chunk embeddings.
    Supports cosine similarity search and filtering.
    """

    def __init__(self):
        self._chunks: Dict[str, DocumentChunk] = {}
        self._embeddings: Dict[str, List[float]] = {}
        self._metadata_index: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

    def add_chunk(self, chunk: DocumentChunk, embedding: List[float]) -> None:
        """Add a chunk with its embedding."""
        self._chunks[chunk.chunk_id] = chunk
        self._embeddings[chunk.chunk_id] = embedding

        # Index metadata
        for key, value in chunk.metadata.items():
            if isinstance(value, str):
                self._metadata_index[key][value].add(chunk.chunk_id)

    def search(self, query_embedding: List[float],
               top_k: int = 10,
               filters: Optional[Dict[str, str]] = None) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar chunks using cosine similarity."""
        scores = []

        for chunk_id, embedding in self._embeddings.items():
            # Apply filters
            if filters:
                match = True
                for key, value in filters.items():
                    if chunk_id in self._metadata_index.get(key, {}).get(value, set()):
                        continue
                    chunk = self._chunks.get(chunk_id)
                    if not chunk or chunk.metadata.get(key) != value:
                        match = False
                        break
                if not match:
                    continue

            similarity = EmbeddingEngine().cosine_similarity(query_embedding, embedding)
            scores.append((chunk_id, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for chunk_id, score in scores[:top_k]:
            if chunk_id in self._chunks:
                results.append((self._chunks[chunk_id], score))

        return results

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        return self._chunks.get(chunk_id)

    def count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()
        self._embeddings.clear()
        self._metadata_index.clear()

    def to_dict(self) -> Dict:
        return {
            'num_chunks': len(self._chunks),
            'num_embeddings': len(self._embeddings),
        }


# ═══════════════════════════════════════════════════════════════════════
# IV. RAG ENGINE
# ═══════════════════════════════════════════════════════════════════════

class RAGEngine:
    """
    Autonomous Retrieval-Augmented Generation pipeline.
    Crawls documents, chunks them, embeds them, and enables semantic retrieval.
    """

    def __init__(self):
        self.chunker = DocumentChunker(chunk_size=512, overlap=64)
        self.embedder = EmbeddingEngine(dimensions=128)
        self.vector_store = VectorStore()
        self.documents: Dict[str, Document] = {}
        self._crawlers: Dict[str, Callable] = {}
        self._is_fitted = False

    def register_crawler(self, name: str, crawler_fn: Callable[[str], str]) -> None:
        """Register a crawler function. Takes a source URI, returns text content."""
        self._crawlers[name] = crawler_fn

    def ingest_text(self, text: str, source: str = "direct_input",
                    title: str = "", metadata: Optional[Dict] = None) -> Document:
        """Ingest raw text as a document."""
        doc = Document(
            content=text,
            source=source,
            title=title or f"Doc {len(self.documents) + 1}",
            metadata=metadata or {},
        )
        return self._process_document(doc)

    def ingest_file(self, filepath: str) -> Optional[Document]:
        """Ingest a file from the filesystem."""
        path = Path(filepath)
        if not path.exists():
            return None

        try:
            content = path.read_text(encoding='utf-8', errors='replace')
            doc = Document(
                content=content,
                source=str(path.absolute()),
                title=path.name,
                metadata={'filepath': str(path), 'extension': path.suffix},
            )
            return self._process_document(doc)
        except Exception as e:
            return None

    def ingest_directory(self, directory: str,
                         pattern: str = "*.txt,*.md,*.py,*.json,*.csv,*.html",
                         recursive: bool = True) -> List[Document]:
        """Ingest all matching files in a directory."""
        patterns = [p.strip() for p in pattern.split(',')]
        docs = []
        path = Path(directory)

        for pat in patterns:
            if recursive:
                files = path.rglob(pat)
            else:
                files = path.glob(pat)
            for f in files:
                if f.is_file():
                    doc = self.ingest_file(str(f))
                    if doc:
                        docs.append(doc)
        return docs

    def ingest_url(self, url: str) -> Optional[Document]:
        """Ingest content from a URL using registered crawlers."""
        for name, crawler in self._crawlers.items():
            try:
                content = crawler(url)
                if content:
                    doc = Document(
                        content=content,
                        source=url,
                        title=f"Web: {url}",
                        metadata={'url': url, 'crawler': name},
                    )
                    return self._process_document(doc)
            except Exception:
                continue

        # Fallback: try basic urllib
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=10) as resp:
                content = resp.read().decode('utf-8', errors='replace')
                # Strip HTML tags for plain text extraction
                content = re.sub(r'<[^>]+>', ' ', content)
                content = re.sub(r'\s+', ' ', content).strip()
                if content:
                    doc = Document(
                        content=content[:100000],  # Limit to 100k chars
                        source=url,
                        title=f"Web: {url}",
                        metadata={'url': url},
                    )
                    return self._process_document(doc)
        except Exception:
            pass

        return None

    def fit(self) -> 'RAGEngine':
        """Fit the embedding model on all ingested documents."""
        all_texts = [doc.content for doc in self.documents.values()]
        if all_texts:
            self.embedder.fit(all_texts)
            # Re-embed all chunks
            self.vector_store.clear()
            for doc in self.documents.values():
                for chunk in doc.chunks:
                    embedding = self.embedder.embed(chunk.content)
                    self.vector_store.add_chunk(chunk, embedding)
            self._is_fitted = True
        return self

    def retrieve(self, query: str, top_k: int = 10,
                 filters: Optional[Dict[str, str]] = None) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve relevant chunks for a query."""
        if not self._is_fitted:
            return []

        query_embedding = self.embedder.embed(query)
        return self.vector_store.search(query_embedding, top_k=top_k, filters=filters)

    def retrieve_context(self, query: str, max_chunks: int = 5,
                         max_tokens: int = 2000) -> str:
        """Retrieve and format context for a query, suitable for LLM or reasoning."""
        results = self.retrieve(query, top_k=max_chunks)
        if not results:
            return ""

        context_parts = []
        total_tokens = 0
        tokenizer = Tokenizer()

        for chunk, score in results:
            tokens = tokenizer(chunk.content)
            if total_tokens + len(tokens) > max_tokens:
                # Truncate this chunk
                remaining = max_tokens - total_tokens
                if remaining > 0:
                    truncated = ' '.join(tokens[:remaining])
                    context_parts.append(f"[source: {chunk.doc_id}] {truncated}")
                break
            context_parts.append(f"[source: {chunk.doc_id}] {chunk.content}")
            total_tokens += len(tokens)

        return '\n\n'.join(context_parts)

    def get_statistics(self) -> Dict[str, Any]:
        """Get RAG pipeline statistics."""
        return {
            'documents': len(self.documents),
            'chunks': self.vector_store.count(),
            'embedding_dimensions': self.embedder.dimensions,
            'vocab_size': len(self.embedder._vocab) if self._is_fitted else 0,
            'is_fitted': self._is_fitted,
            'crawlers': list(self._crawlers.keys()),
        }

    def _process_document(self, doc: Document) -> Document:
        """Process a document through the chunking pipeline."""
        # Chunk the document
        self.chunker.chunk_recursive(doc)

        # Embed chunks if fitted
        if self._is_fitted:
            for chunk in doc.chunks:
                embedding = self.embedder.embed(chunk.content)
                self.vector_store.add_chunk(chunk, embedding)

        self.documents[doc.doc_id] = doc
        return doc

    def to_dict(self) -> Dict:
        return {
            'statistics': self.get_statistics(),
            'documents': {k: v.to_dict() for k, v in self.documents.items()},
        }