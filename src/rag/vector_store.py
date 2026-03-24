"""
Build and manage the ChromaDB vector store for the RAG system.
Embeds manual chunks and fault case descriptions for similarity search.
"""

import json
import yaml
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_fault_case_documents(fault_cases):
    """Convert structured fault cases into text documents for embedding."""
    docs = []
    for case in fault_cases:
        text = (
            f"Fault Type: {case['fault_type']}\n"
            f"Severity: {case['severity']}\n"
            f"Location: {case['location']}\n"
            f"Symptoms: {case['symptoms']}\n"
            f"Spectrogram Pattern: {case['spectrogram_pattern']}\n"
            f"Root Cause: {case['root_cause']}\n"
            f"Recommended Action: {case['recommended_action']}\n"
            f"Similar Cases: {case['similar_cases']}"
        )
        docs.append({
            "id": f"fault_{case['fault_type']}",
            "text": text,
            "metadata": {
                "source_type": "fault_case",
                "fault_type": case["fault_type"],
                "severity": case["severity"],
            },
        })
    return docs


def build_vector_store(config_path="configs/config.yaml"):
    cfg = load_config(config_path)
    rag_cfg = cfg["rag"]

    manual_chunks_path = Path(cfg["paths"]["manual_chunks"])
    fault_cases_path = Path(cfg["paths"]["fault_cases"])
    chroma_dir = Path(cfg["paths"]["chroma_db"])

    # Load data
    with open(manual_chunks_path, "r", encoding="utf-8") as f:
        manual_chunks = json.load(f)
    with open(fault_cases_path, "r", encoding="utf-8") as f:
        fault_cases = json.load(f)

    print(f"Loaded {len(manual_chunks)} manual chunks, {len(fault_cases)} fault cases")

    # Embedding model
    print(f"Loading embedding model: {rag_cfg['embedding_model']}")
    embedder = SentenceTransformer(rag_cfg["embedding_model"])

    # Initialize ChromaDB
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    # Delete existing collections if they exist (for rebuilding)
    for name in ["manual_chunks", "fault_cases"]:
        try:
            client.delete_collection(name)
        except ValueError:
            pass

    # --- Manual chunks collection ---
    manual_collection = client.create_collection(
        name="manual_chunks",
        metadata={"description": "Rolling bearing handbook text chunks"},
    )
    manual_texts = [c["text"] for c in manual_chunks]
    manual_ids = [f"manual_{c['id']}" for c in manual_chunks]
    manual_metadatas = [
        {"source_type": "manual", "page": c["page"], "source": c["source"]}
        for c in manual_chunks
    ]

    # Embed and add in batches
    batch_size = 64
    print("Embedding manual chunks...")
    for i in range(0, len(manual_texts), batch_size):
        batch_texts = manual_texts[i : i + batch_size]
        batch_ids = manual_ids[i : i + batch_size]
        batch_meta = manual_metadatas[i : i + batch_size]
        batch_embeddings = embedder.encode(batch_texts).tolist()
        manual_collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=batch_embeddings,
            metadatas=batch_meta,
        )
    print(f"  Added {len(manual_texts)} manual chunks to vector store")

    # --- Fault cases collection ---
    fault_docs = build_fault_case_documents(fault_cases)
    fault_collection = client.create_collection(
        name="fault_cases",
        metadata={"description": "Bearing fault case knowledge base"},
    )
    fault_texts = [d["text"] for d in fault_docs]
    fault_ids = [d["id"] for d in fault_docs]
    fault_metadatas = [d["metadata"] for d in fault_docs]
    fault_embeddings = embedder.encode(fault_texts).tolist()

    fault_collection.add(
        ids=fault_ids,
        documents=fault_texts,
        embeddings=fault_embeddings,
        metadatas=fault_metadatas,
    )
    print(f"  Added {len(fault_texts)} fault cases to vector store")
    print(f"Vector store persisted to {chroma_dir}")


def get_retriever(config_path="configs/config.yaml"):
    """Load the persisted vector store and return a retriever-like interface."""
    cfg = load_config(config_path)
    rag_cfg = cfg["rag"]
    chroma_dir = Path(cfg["paths"]["chroma_db"])

    client = chromadb.PersistentClient(path=str(chroma_dir))
    embedder = SentenceTransformer(rag_cfg["embedding_model"])

    manual_collection = client.get_collection("manual_chunks")
    fault_collection = client.get_collection("fault_cases")

    return VectorRetriever(embedder, manual_collection, fault_collection, rag_cfg["top_k"])


class VectorRetriever:
    """Simple retriever that queries both manual and fault case collections."""

    def __init__(self, embedder, manual_collection, fault_collection, top_k=5):
        self.embedder = embedder
        self.manual_collection = manual_collection
        self.fault_collection = fault_collection
        self.top_k = top_k

    def retrieve(self, query, source_type=None, top_k=None):
        """Retrieve relevant documents.

        Args:
            query: search query string
            source_type: "manual", "fault_case", or None (searches both)
            top_k: override default top_k
        """
        k = top_k or self.top_k
        query_embedding = self.embedder.encode([query]).tolist()
        results = []

        if source_type in (None, "manual"):
            manual_results = self.manual_collection.query(
                query_embeddings=query_embedding, n_results=k
            )
            for doc, meta, dist in zip(
                manual_results["documents"][0],
                manual_results["metadatas"][0],
                manual_results["distances"][0],
            ):
                results.append({"text": doc, "metadata": meta, "distance": dist})

        if source_type in (None, "fault_case"):
            fault_results = self.fault_collection.query(
                query_embeddings=query_embedding, n_results=min(k, 3)
            )
            for doc, meta, dist in zip(
                fault_results["documents"][0],
                fault_results["metadatas"][0],
                fault_results["distances"][0],
            ):
                results.append({"text": doc, "metadata": meta, "distance": dist})

        # Sort by distance (lower = more similar)
        results.sort(key=lambda x: x["distance"])
        return results[:k]


if __name__ == "__main__":
    build_vector_store()
