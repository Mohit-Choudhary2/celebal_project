"""RAG (Retrieval-Augmented Generation) Engine for Python/Pandas error recovery.
Provides fast semantic and keyword retrieval over official documentation,
error causes, and verified syntax fixes.
"""

import math
import re
from typing import List, Dict, Any, Tuple
from backend.rag_knowledge_base import OFFICIAL_DOC_CHUNKS


class RAGEngine:
    """Retrieves relevant official Python/Pandas/Plotly documentation

    to guide code correction and self-healing.
    """

    def __init__(self, doc_chunks: List[Dict[str, Any]] = None):
        self.doc_chunks = doc_chunks or OFFICIAL_DOC_CHUNKS
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        """Normalize and tokenize text into words and alphanumeric tokens."""
        text = text.lower()
        tokens = re.findall(r'[a-zA-Z0-9_\.]+', text)
        return [t for t in tokens if len(t) > 1]

    def _build_index(self):
        """Constructs an inverted index and TF-IDF vectors for fast retrieval."""
        self.doc_tokens: List[List[str]] = []
        self.df_counts: Dict[str, int] = {}
        total_docs = len(self.doc_chunks)

        for doc in self.doc_chunks:
            full_text = " ".join([
                doc.get("title", ""),
                doc.get("category", ""),
                " ".join(doc.get("error_types", [])),
                " ".join(doc.get("keywords", [])),
                doc.get("doc_content", ""),
                doc.get("example_bad", ""),
                doc.get("example_good", "")
            ])
            tokens = self._tokenize(full_text)
            self.doc_tokens.append(tokens)

            # Unique terms in this document for document frequency
            unique_terms = set(tokens)
            for term in unique_terms:
                self.df_counts[term] = self.df_counts.get(term, 0) + 1

        # Calculate IDF
        self.idf: Dict[str, float] = {}
        for term, count in self.df_counts.items():
            self.idf[term] = math.log((1 + total_docs) / (1 + count)) + 1.0

        # Precompute TF-IDF vector norms for docs
        self.doc_vectors: List[Dict[str, float]] = []
        self.doc_norms: List[float] = []

        for tokens in self.doc_tokens:
            tf: Dict[str, float] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            length = len(tokens) or 1

            vec: Dict[str, float] = {}
            sum_sq = 0.0
            for t, count in tf.items():
                val = (count / length) * self.idf.get(t, 1.0)
                vec[t] = val
                sum_sq += val * val

            self.doc_vectors.append(vec)
            self.doc_norms.append(math.sqrt(sum_sq) or 1.0)

    def retrieve(self, query: str, error_type: str = "", top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve the top-K most relevant documentation chunks based on error message,

        traceback, and failing code snippet.
        """
        query_tokens = self._tokenize(f"{query} {error_type}")
        if not query_tokens:
            return self.doc_chunks[:top_k]

        q_tf: Dict[str, float] = {}
        for t in query_tokens:
            q_tf[t] = q_tf.get(t, 0) + 1
        q_len = len(query_tokens)

        q_vec: Dict[str, float] = {}
        q_sum_sq = 0.0
        for t, count in q_tf.items():
            val = (count / q_len) * self.idf.get(t, 1.0)
            q_vec[t] = val
            q_sum_sq += val * val
        q_norm = math.sqrt(q_sum_sq) or 1.0

        scores: List[Tuple[float, int]] = []
        normalized_error = error_type.lower().strip()

        for idx, (doc_vec, doc_norm) in enumerate(zip(self.doc_vectors, self.doc_norms)):
            # Cosine similarity
            dot_product = sum(q_vec[t] * doc_vec[t] for t in q_vec if t in doc_vec)
            cos_sim = dot_product / (q_norm * doc_norm)

            # Error type exact match boost
            doc = self.doc_chunks[idx]
            doc_errors = [e.lower() for e in doc.get("error_types", [])]
            if normalized_error and any(normalized_error in e or e in normalized_error for e in doc_errors):
                cos_sim += 0.5  # Significant boost for exact error family

            scores.append((cos_sim, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = [self.doc_chunks[idx] for _, idx in scores[:top_k]]
        return results

    def format_rag_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documentation into a clear instruction block for the agent."""
        if not retrieved_docs:
            return "No specific documentation retrieved."

        lines = ["=== RELEVANT OFFICIAL PYTHON/PANDAS DOCUMENTATION & FIX PATTERNS ==="]
        for i, doc in enumerate(retrieved_docs, 1):
            lines.append(f"\n[Doc {i}: {doc['title']} | Category: {doc['category']}]")
            lines.append(doc["doc_content"].strip())
            if doc.get("example_bad"):
                lines.append(f"  ❌ Antipattern: {doc['example_bad']}")
            if doc.get("example_good"):
                lines.append(f"  ✅ Verified fix: {doc['example_good']}")
        lines.append("\n=== END OFFICIAL DOCUMENTATION ===")
        return "\n".join(lines)


# Singleton instance
rag_engine = RAGEngine()
