
import re
from typing import List, Tuple, Dict
import numpy as np # Ensure numpy is installed

class BM25:
    def __init__(self, corpus: List[str], k1: float = 1.2, b: float = 0.75):
        """
        Initializes the BM25 model.

        Args:
            corpus (List[str]): A list of documents (strings) in the corpus.
            k1 (float): Parameter for term frequency scaling.
            b (float): Parameter for document length normalization.
        """
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.avgdl = self._calculate_avgdl()
        self.index = self._build_index()

    def _calculate_avgdl(self) -> float:
        """Calculates the average document length in the corpus."""
        return sum(len(doc.split()) for doc in self.corpus) / len(self.corpus)

    def _build_index(self) -> Dict[str, Dict[str, int]]:
        """Builds an index of term frequencies for each document."""
        index = {}
        for doc_id, doc in enumerate(self.corpus):
            term_freqs = {}
            for term in re.findall(r'\b\w+\b', doc.lower()):  # Simple tokenization
                term_freqs[term] = term_freqs.get(term, 0) + 1
            index[str(doc_id)] = term_freqs
        return index

    def _calculate_idf(self, term: str) -> float:
        """Calculates the inverse document frequency (IDF) for a term."""
        df = sum(1 for doc_id, term_freqs in self.index.items() if term in term_freqs)
        if df == 0:
            return 0.0  # Avoid division by zero
        return np.log(1 + (len(self.corpus) - df + 0.5) / (df + 0.5)) # Use numpy

    def get_scores(self, query: str) -> Dict[str, float]:
        """
        Calculates BM25 scores for a query against the corpus.

        Args:
            query (str): The query string.

        Returns:
            Dict[str, float]: A dictionary of document IDs and their BM25 scores.
        """
        query_terms = re.findall(r'\b\w+\b', query.lower())
        doc_scores = {}
        for doc_id, term_freqs in self.index.items():
            score = 0.0
            doc_len = len(self.corpus[int(doc_id)].split())
            for term in query_terms:
                if term in term_freqs:
                    tf = term_freqs[term]
                    idf = self._calculate_idf(term)
                    score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl)))
            doc_scores[doc_id] = score
        return doc_scores

    def get_top_n(self, query: str, top_n: int = 3) -> List[Tuple[int, float]]:
        """
        Retrieves the top N documents based on BM25 scores.

        Args:
            query (str): The query string.
            top_n (int): The number of top documents to retrieve.

        Returns:
            List[Tuple[int, float]]: A list of (document ID, score) tuples,
                                     sorted in descending order of score.
        """
        scores = self.get_scores(query)
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_scores[:top_n]