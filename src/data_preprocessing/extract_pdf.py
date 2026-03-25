"""
Extract text from the rolling-bearing-handbook PDF and chunk it
into overlapping segments suitable for RAG embedding.
"""

import json
import yaml
from pathlib import Path

import fitz  # PyMuPDF


def load_config(config_path="configs/config.yaml"):
    from pathlib import Path
    config_path = Path(config_path)
    if not config_path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = project_root / config_path
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def extract_pages(pdf_path):
    """Extract text from each page of the PDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages.append({"page": page_num + 1, "text": text.strip()})
    doc.close()
    return pages


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def run(config_path="configs/config.yaml"):
    cfg = load_config(config_path)
    print("[DEBUG] cfg['paths']:", cfg["paths"])
    pdf_path = Path(cfg["paths"]["manual_pdf"])
    out_path = Path(cfg["paths"]["manual_chunks"])

    rag_cfg = cfg["rag"]
    chunk_size = rag_cfg["chunk_size"]
    chunk_overlap = rag_cfg["chunk_overlap"]

    print(f"Extracting text from {pdf_path} ...")
    pages = extract_pages(str(pdf_path))
    print(f"  Extracted text from {len(pages)} pages")

    all_chunks = []
    chunk_id = 0
    for page_info in pages:
        page_chunks = chunk_text(page_info["text"], chunk_size, chunk_overlap)
        for chunk in page_chunks:
            all_chunks.append({
                "id": chunk_id,
                "page": page_info["page"],
                "text": chunk,
                "source": "rolling-bearing-handbook",
            })
            chunk_id += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(all_chunks)} chunks to {out_path}")


if __name__ == "__main__":
    run()
