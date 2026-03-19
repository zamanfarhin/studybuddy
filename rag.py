"""
RAG module: PDF processing, chunking, embeddings, and ChromaDB storage.
"""

import fitz  # PyMuPDF
import tiktoken
from openai import OpenAI
import chromadb
from chromadb.config import Settings
import os
import hashlib


class RAGEngine:
    def __init__(self, openai_api_key: str, persist_directory: str = "./chroma_db"):
        self.client = OpenAI(api_key=openai_api_key, timeout=30.0)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.chunk_size = 512
        self.chunk_overlap = 50

        # Initialize ChromaDB with persistence
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name="studybuddy_docs",
            metadata={"hnsw:space": "cosine"}
        )

    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes using PyMuPDF."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))

    def chunk_text(self, text: str) -> list[str]:
        """Chunk text into pieces of ~512 tokens with 50 token overlap."""
        tokens = self.tokenizer.encode(text)
        chunks = []

        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            start = end - self.chunk_overlap

        return chunks

    def get_embedding(self, text: str) -> list[float]:
        """Get embedding for text using OpenAI text-embedding-3-small."""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            timeout=30.0
        )
        return response.data[0].embedding

    def generate_chunk_id(self, doc_name: str, chunk_index: int) -> str:
        """Generate a unique ID for a chunk."""
        return hashlib.md5(f"{doc_name}_{chunk_index}".encode()).hexdigest()

    def process_and_store_pdf(self, pdf_bytes: bytes, filename: str) -> int:
        """Process PDF, chunk it, embed chunks, and store in ChromaDB."""
        print(f"[RAG] Extracting text from '{filename}'...")
        text = self.extract_text_from_pdf(pdf_bytes)
        print(f"[RAG] Extracted {len(text)} chars. Chunking with chunk_size={self.chunk_size}...")

        chunks = self.chunk_text(text)
        print(f"[RAG] Created {len(chunks)} chunks. Embedding each chunk (30s timeout per call)...")

        if not chunks:
            print("[RAG] No chunks found — PDF may be empty or image-only.")
            return 0

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            print(f"[RAG]   Embedding chunk {i + 1}/{len(chunks)}...", flush=True)
            chunk_id = self.generate_chunk_id(filename, i)
            embedding = self.get_embedding(chunk)

            ids.append(chunk_id)
            embeddings.append(embedding)
            documents.append(chunk)
            metadatas.append({"source": filename, "chunk_index": i})

        print(f"[RAG] Storing {len(chunks)} chunks in ChromaDB...")
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"[RAG] Done storing chunks for '{filename}'.")

        return len(chunks)

    def retrieve(self, query: str, n_results: int = 3) -> list[dict]:
        """Retrieve relevant chunks for a query."""
        if self.collection.count() == 0:
            return []

        query_embedding = self.get_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.collection.count())
        )

        retrieved = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                retrieved.append({
                    "content": doc,
                    "source": results["metadatas"][0][i]["source"],
                    "distance": results["distances"][0][i] if results["distances"] else None
                })

        return retrieved

    def get_document_count(self) -> int:
        """Get the number of chunks stored."""
        return self.collection.count()

    def clear_collection(self):
        """Clear all documents from the collection."""
        self.chroma_client.delete_collection("studybuddy_docs")
        self.collection = self.chroma_client.get_or_create_collection(
            name="studybuddy_docs",
            metadata={"hnsw:space": "cosine"}
        )
