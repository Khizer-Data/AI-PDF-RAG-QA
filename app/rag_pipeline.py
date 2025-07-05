# rag_pipeline.py

import fitz  # PyMuPDF
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import logging

class RAGPipeline:
    def __init__(self):
        self.embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.chunks: List[str] = []
        self.metadata: List[Dict] = []
        self.indexed_files: List[str] = []
        self.detected_domain = "General"

    def load_and_index_pdfs(self, file_paths: List[str]):
        self.chunks = []
        self.metadata = []
        self.indexed_files = file_paths
        all_text = ""

        for file_path in file_paths:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            all_text += text
            self.chunks.extend(self.chunk_text(text, file_path))

        texts = [chunk['text'] for chunk in self.chunks]
        vectors = self.embeddings_model.encode(texts, batch_size=32, show_progress_bar=True)
        vectors = np.array(vectors).astype("float32")
        faiss.normalize_L2(vectors)  # Normalize for cosine similarity

        self.index = faiss.IndexFlatIP(vectors.shape[1])  # Use cosine
        self.index.add(vectors)
        self.metadata = self.chunks

        self.detected_domain = self.detect_domain(all_text)

    def chunk_text(self, text: str, source: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append({"text": chunk, "source": source})
        return chunks

    def detect_domain(self, text: str) -> str:
        text_lower = text.lower()
        if "resume" in text_lower or "curriculum vitae" in text_lower:
            return "Resume"
        elif "contract" in text_lower or "clause" in text_lower or "agreement" in text_lower:
            return "Legal"
        elif "policy" in text_lower or "compliance" in text_lower or "regulation" in text_lower:
            return "Company Policy"
        elif "invoice" in text_lower or "balance" in text_lower or "statement" in text_lower:
            return "Finance"
        elif "diagnosis" in text_lower or "patient" in text_lower or "medical" in text_lower:
            return "Medical"
        else:
            return "General"

    def query(self, question: str, top_k: int = 3) -> List[Dict]:
        question_vec = self.embeddings_model.encode([question]).astype("float32")
        faiss.normalize_L2(question_vec)
        if self.index is None:
            return []
        _, indices = self.index.search(question_vec, top_k)
        return [self.metadata[i] for i in indices[0] if i < len(self.metadata)]

    def cleanup(self):
        for path in self.indexed_files:
            try:
                os.remove(path)
            except Exception as e:
                print(f"Failed to remove {path}: {e}")
