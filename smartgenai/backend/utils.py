
import fitz  # PyMuPDF
import os
import re
import io
import matplotlib.pyplot as plt
import networkx as nx
from wordcloud import WordCloud
import base64
from typing import Dict, Tuple, List
import json
from hashlib import sha256
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np

# Helper functions

def sanitize_filename(filename: str) -> str:
    """Sanitizes a filename to remove potentially harmful characters."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", filename)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file using PyMuPDF."""
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def calculate_top_terms(text: str, top_n: int = 10) -> Dict[str, float]:
    """Calculates the top N most frequent terms in a text."""
    words = re.findall(r'\b\w+\b', text.lower())  # Simple tokenization
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    sorted_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    top_terms = dict(sorted_counts[:top_n])
    return top_terms

def create_bar_chart(top_terms: Dict[str, float]) -> bytes:
    """Creates a bar chart of the top terms and returns it as a PNG image."""
    if not top_terms:
        return None

    terms = list(top_terms.keys())
    counts = list(top_terms.values())

    plt.figure(figsize=(10, 5))
    plt.bar(terms, counts)
    plt.xlabel("Terms")
    plt.ylabel("Frequency")
    plt.title("Top Terms")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()  # Close the plot to free memory
    return img.getvalue()

def create_pie_chart(top_terms: Dict[str, float]) -> bytes:
    """Creates a pie chart of the top terms and returns it as a PNG image."""
    if not top_terms:
        return None

    terms = list(top_terms.keys())
    counts = list(top_terms.values())

    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=terms, autopct='%1.1f%%', startangle=140)
    plt.title("Top Terms Distribution")
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    return img.getvalue()

def get_file_extension(filename: str) -> str:
    """Gets the file extension from a filename."""
    return os.path.splitext(filename)[1].lower()

def generate_unique_id(data: str) -> str:
    """Generates a unique ID from a string."""
    return sha256(data.encode()).hexdigest()

def get_file_path(doc_id: str, file_name: str, base_dir: str = "static/uploads") -> str:
    """Generates the full file path for a document's file."""
    os.makedirs(base_dir, exist_ok=True)  # Ensure the directory exists
    file_name = sanitize_filename(file_name)
    file_path = os.path.join(base_dir, doc_id, file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    return file_path

def get_image_as_base64(image_path: str) -> str:
    """Encodes an image file as a base64 string."""
    try:
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
        return base64.b64encode(img_data).decode('utf-8')
    except FileNotFoundError:
        print(f"Image not found: {image_path}")
        return ""
    except Exception as e:
        print(f"Error reading image: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Tuple[str, str]]:
    """
    Chunks text into smaller pieces with overlap.
    Returns a list of (chunk_id, text) tuples.
    """
    chunks = []
    words = text.split()
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunk_id = generate_unique_id(chunk)
        chunks.append((chunk_id, chunk))
        start += (chunk_size - overlap)
    return chunks

def calculate_bm25_scores(query: str, index: dict, k1: float = 1.2, b: float = 0.75) -> Dict[str, float]:
    """
    Calculates BM25 scores for a query against a pre-built index.
    Returns a dictionary of chunk_id: score.
    """
    query_terms = re.findall(r'\b\w+\b', query.lower())
    doc_scores = {}
    avg_doc_len = index["avg_doc_len"]
    N = index["N"]

    for chunk_id, chunk_data in index["index"].items():
        score = 0.0
        doc_len = len(re.findall(r'\b\w+\b', query.lower()))
        for term in query_terms:
            if term in chunk_data["term_freq"]:
                tf = chunk_data["term_freq"][term]
                df = index["df"].get(term, 0)  # Document frequency
                if df == 0:
                    continue  # Skip terms not in the index
                idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
                score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
        doc_scores[chunk_id] = score

    return doc_scores

def build_bm25_index(chunks: List[Tuple[str, str]]) -> dict:
    """
    Builds a BM25 index from a list of (chunk_id, text) tuples.
    """
    index = {}
    doc_lens = []
    term_dfs = {}  # Term document frequencies
    for chunk_id, text in chunks:
        doc_len = len(re.findall(r'\b\w+\b', text.lower())) #words count
        doc_lens.append(doc_len)
        term_freq = {}
        for term in re.findall(r'\b\w+\b', text.lower()):
            term_freq[term] = term_freq.get(term, 0) + 1 # count terms in each chunk
        index[chunk_id] = {"doc_len": doc_len, "term_freq": term_freq} # chunk data
        for term in term_freq:
            term_dfs[term] = term_dfs.get(term, 0) + 1

    avg_doc_len = sum(doc_lens) / len(doc_lens)
    N = len(chunks)  # Number of documents (chunks)

    return {
        "index": index,
        "avg_doc_len": avg_doc_len,
        "N": N,
        "df": term_dfs,
    }


def generate_mind_map(text: str, doc_id: str) -> bytes:
    """
    Generates a mind map visualization from a given text using networkx and matplotlib.
    Returns a PNG image as bytes.
    """
    try:
        # Simple keyword extraction (improve this with more sophisticated methods)
        keywords = [word for word in re.findall(r'\b\w+\b', text.lower()) if len(word) > 3]
        # Create a graph
        graph = nx.Graph()
        graph.add_node("Main Topic")
        for keyword in keywords[:10]:  # Limit keywords for a cleaner visualization
            graph.add_edge("Main Topic", keyword)

        # Draw the graph
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(graph)  # You can try other layout algorithms
        nx.draw(graph, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10, font_weight="bold")
        plt.title("Recall Mind Map")

        # Save to image
        img = io.BytesIO()
        plt.savefig(img, format="png")
        plt.close()
        return img.getvalue()

    except Exception as e:
        print(f"Error generating mind map: {e}")
        return None

def generate_power_hour_content(text: str) -> Dict[str, str]:
    """
    Generates Power Hour content: notes, formulas, and top 10 terms.
    This is a placeholder and needs to be improved with actual content generation.
    Returns a dictionary with content sections.
    """
    top_terms = calculate_top_terms(text)
    notes = "Detailed notes will be generated here based on the document content.  This is a placeholder."
    formulas = "Key formulas will be extracted and summarized here.  This is a placeholder."
    top_10_terms = ", ".join(top_terms.keys())

    return {
        "notes": notes,
        "formulas": formulas,
        "top_10_terms": top_10_terms,
    }

def get_static_image_path(image_name: str) -> str:
    """Returns the path to a static image."""
    return os.path.join("static", image_name) # Assumes static dir at root

def get_static_image_as_base64(image_name: str) -> str:
    """Retrieves a static image and returns it as a base64 encoded string"""
    image_path = get_static_image_path(image_name)
    return get_image_as_base64(image_path)