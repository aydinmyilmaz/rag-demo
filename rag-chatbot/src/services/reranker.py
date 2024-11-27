import os
from typing import List, Dict, Tuple
import numpy as np
from collections import Counter
from math import log
import streamlit as st

class BM25Reranker:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_freqs = Counter()
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.N = 0

    def preprocess(self, text: str) -> List[str]:
        return text.lower().split()

    def fit(self, documents: List[str]):
        self.corpus = [self.preprocess(doc) for doc in documents]
        self.N = len(self.corpus)

        self.doc_freqs = Counter()
        for doc in self.corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                self.doc_freqs[term] += 1

        self.doc_lengths = [len(doc) for doc in self.corpus]
        self.avg_doc_length = sum(self.doc_lengths) / self.N if self.N > 0 else 0

        # Log corpus statistics
        st.write("🔍 BM25 Corpus Statistics:")
        st.write(f"- Number of documents: {self.N}")
        st.write(f"- Average document length: {self.avg_doc_length:.2f} terms")
        st.write(f"- Vocabulary size: {len(self.doc_freqs)} unique terms")

    def score(self, query: str, doc: str) -> float:
        query_terms = self.preprocess(query)
        doc_terms = self.preprocess(doc)
        doc_len = len(doc_terms)

        doc_term_freqs = Counter(doc_terms)

        score = 0.0
        term_scores = []  # Track individual term contributions

        for term in query_terms:
            if term in self.doc_freqs:
                idf = log((self.N - self.doc_freqs[term] + 0.5) /
                         (self.doc_freqs[term] + 0.5) + 1.0)

                tf = doc_term_freqs[term]
                norm_tf = ((tf * (self.k1 + 1)) /
                          (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)))

                term_score = idf * norm_tf
                term_scores.append((term, term_score))
                score += term_score

        return score, term_scores

    def rerank(self, query: str, documents: List[Dict], top_k: int = None) -> List[Dict]:
        if not documents:
            return documents

        if top_k is None:
            top_k = len(documents)

        # Extract document contents
        doc_contents = [doc['content'] for doc in documents]

        # Log initial order
        st.write("\n📝 Initial Document Order:")
        for i, doc in enumerate(documents):
            preview = doc['content'][:100] + "..." if len(doc['content']) > 100 else doc['content']
            st.write(f"{i+1}. Page {doc['page_number']}: {preview}")

        # Fit BM25 on the retrieved documents
        self.fit(doc_contents)

        # Score documents with detailed logging
        scores_with_details = []
        for i, doc in enumerate(doc_contents):
            score, term_scores = self.score(query, doc)
            scores_with_details.append((i, score, term_scores))

        # Sort by scores
        sorted_indices = [i for i, score, _ in sorted(scores_with_details,
                                                    key=lambda x: x[1],
                                                    reverse=True)]

        # Log reranking details
        st.write("\n📊 BM25 Reranking Details:")
        st.write(f"Query terms: {self.preprocess(query)}")

        for rank, idx in enumerate(sorted_indices[:top_k]):
            orig_pos = idx + 1
            doc = documents[idx]
            score = scores_with_details[idx][1]
            term_scores = scores_with_details[idx][2]

            st.write(f"\nRank {rank+1} (was {orig_pos}):")
            st.write(f"- Page {doc['page_number']}")
            st.write(f"- BM25 Score: {score:.4f}")
            st.write("- Term contributions:")
            for term, term_score in term_scores:
                st.write(f"  • {term}: {term_score:.4f}")

            preview = doc['content'][:100] + "..." if len(doc['content']) > 100 else doc['content']
            st.write(f"- Content: {preview}")

        # Return reranked documents
        reranked_docs = [documents[i] for i in sorted_indices[:top_k]]
        return reranked_docs