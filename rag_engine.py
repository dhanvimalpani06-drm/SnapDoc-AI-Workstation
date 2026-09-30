"""
SnapDoc AI — On-Device RAG Engine
Lightweight TF-IDF & Cosine Similarity Vector Store.
Bypasses PyArrow/Torch DLLs to run under strict Windows security policies.
"""

import math
import re
import importlib
import numpy as np

class DocumentProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_pdf(self, pdf_file):
        try:
            pypdf = importlib.import_module("pypdf")
        except ImportError as exc:
            raise RuntimeError(
                "PDF extraction requires the 'pypdf' package. "
                "Install it with: pip install pypdf"
            ) from exc

        reader = pypdf.PdfReader(pdf_file)
        metadata = []
        full_text = ""
        
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            full_text += text + "\n"
            metadata.append({"page_number": idx + 1, "char_count": len(text)})
            
        return full_text, metadata

    def create_chunks(self, text):
        words = text.split()
        if not words:
            return []
            
        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
                
        return chunks if chunks else [text]


class VectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.chunks = []
        self.vocabulary = {}
        self.idf = {}
        self.embeddings = None

    def _tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())

    def index_document(self, chunks):
        self.chunks = chunks
        if not chunks:
            return
            
        doc_count = len(chunks)
        tokenized_chunks = [self._tokenize(c) for c in chunks]
        
        vocab = {}
        doc_freq = {}
        
        for tokens in tokenized_chunks:
            seen = set()
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
                if token not in seen:
                    doc_freq[token] = doc_freq.get(token, 0) + 1
                    seen.add(token)
                    
        self.vocabulary = vocab
        self.idf = {word: math.log((doc_count + 1) / (freq + 1)) + 1.0 for word, freq in doc_freq.items()}
        
        matrix = np.zeros((doc_count, max(1, len(vocab))), dtype=np.float32)
        for i, tokens in enumerate(tokenized_chunks):
            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            for token, count in tf.items():
                if token in vocab:
                    col_idx = vocab[token]
                    matrix[i, col_idx] = count * self.idf[token]
                    
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = matrix / norms

    def search(self, query, top_k=3):
        if self.embeddings is None or len(self.chunks) == 0 or not self.vocabulary:
            return []
            
        tokens = self._tokenize(query)
        query_vec = np.zeros((1, len(self.vocabulary)), dtype=np.float32)
        
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
            
        for token, count in tf.items():
            if token in self.vocabulary:
                col_idx = self.vocabulary[token]
                query_vec[0, col_idx] = count * self.idf.get(token, 1.0)
                
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        scores = np.dot(self.embeddings, query_vec.T).squeeze()
        if scores.ndim == 0:
            scores = np.array([scores])

        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "chunk_id": int(idx),
                "score": float(scores[idx]),
                "content": self.chunks[idx]
            })
            
        return results