# Industrial Fault Diagnosis — Project Overview

This document describes the repository: purpose, architecture, key files and functions, run instructions, LLM/RAG details, and troubleshooting notes.

## Purpose
This project is an end-to-end system for rolling-bearing fault diagnosis combining:
- A CNN classifier (spectrogram-based) for initial fault classification.
- A Retrieval-Augmented Generation (RAG) pipeline that searches a handbook + fault-case knowledge base, then uses a local LLM (Ollama) to produce a structured diagnosis and recommendations.
- A Streamlit UI for diagnosis, knowledge-base Q&A, case browsing, and dashboard.

## Quick start
1. Create and activate a Python virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Configure environment variables (optional): create `.env` with

```
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b
```

3. Ensure Ollama is installed and required model(s) are available locally:

```powershell
# Install ollama (system package) and optionally the python client
# Then pull the model you want to use, e.g.:
ollama pull llama3.2:3b
ollama list
```

4. Run the Streamlit UI:

```powershell
streamlit run src/app/main.py
```

## High-level architecture
- `src/app/main.py` — Streamlit application. Contains pages:
  - Diagnose (spectrogram / .mat -> CNN -> RAG report)
  - Knowledge Base (conversational Q&A)
  - Case History (fault-case browser)
  - Dashboard

- `src/rag/` — RAG components
  - `retrieval_chain.py` — LLM initialization (`get_llm()`), `RetrievalChain` class, prompt templates.
  - `vector_store.py` — ChromaDB persistent vector store builder and `VectorRetriever` (embeddings via SentenceTransformers).
  - `diagnosis_pipeline.py` — `DiagnosisPipeline` that wires CNN + retriever + LLM and exposes `diagnose_from_image`, `diagnose_from_signal`, `ask_manual`, and `classify_image`.

- `src/models/` — model code
  - `cnn_classifier.py` — `BearingFaultCNN` and related utilities.
  - `train.py` — training script for CNN (produce a checkpoint saved under config path).

- `src/data_preprocessing/` — spectrogram generation, PDF/text extraction, and fault-case building (`generate_spectrograms.py`, `build_fault_cases.py`, etc.).

- `configs/config.yaml` and `src/configs/config.yaml` — central configuration (paths, rag settings, embedding model name).

- `data/` — stores `fault_cases.json`, `manual_chunks.json`, and a `chroma_db/` directory (persisted Chroma DB).

## Key files and exported functions (quick reference)
- `src/app/main.py` ([src/app/main.py](src/app/main.py))
  - `main()` — entrypoint for Streamlit UI.
  - `page_diagnose()` — UI + calls to `DiagnosisPipeline`.
  - `page_knowledge_base()` — chat UI backed by `DiagnosisPipeline.ask_manual()`.

- `src/rag/retrieval_chain.py` ([src/rag/retrieval_chain.py](src/rag/retrieval_chain.py))
  - `load_config(config_path)` — loads YAML config.
  - `get_llm(config_path)` — returns a configured Ollama LLM client. Behavior:
    - Reads `OLLAMA_MODEL` env var or `rag.ollama_model` from config.
    - Validates installed Ollama models using `ollama ls` (CLI) and/or `ollama.Client()` (python client).
    - Matches requested model to an installed candidate (substring/weak match) and returns the LangChain Ollama client with that model.
  - `RetrievalChain` class:
    - `diagnose(query, fault_class=None, confidence=None)` — retrieve context via retriever and call LLM to produce a structured diagnosis using `DIAGNOSIS_PROMPT_TEMPLATE`.
    - `ask(question)` — retrieve manual-only context and call LLM with `QA_PROMPT_TEMPLATE`.

- `src/rag/vector_store.py` ([src/rag/vector_store.py](src/rag/vector_store.py))
  - `load_config(config_path)` — load YAML.
  - `build_vector_store(config_path)` — build or rebuild persistent Chroma collections (`manual_chunks` and `fault_cases`) by embedding with `SentenceTransformer`.
  - `get_retriever(config_path)` — returns `VectorRetriever` instance.
  - `VectorRetriever.retrieve(query, source_type=None, top_k=None)` — returns sorted results with document text, metadata and distances.

- `src/rag/diagnosis_pipeline.py` ([src/rag/diagnosis_pipeline.py](src/rag/diagnosis_pipeline.py))
  - `DiagnosisPipeline(config_path)` — orchestrator class.
    - `_load_cnn()` — loads CNN from `cfg['paths']['cnn_model']` and prepares transforms.
    - `_load_rag()` — gets retriever and LLM and constructs `RetrievalChain`.
    - `classify_image(image_path)` — runs the CNN and returns predicted class, confidence, top-3.
    - `diagnose_from_image(image_path, user_query=None)` — classify then call RAG retrieval + LLM.
    - `diagnose_from_signal(mat_file_path, user_query=None)` — extract DE signal, generate spectrogram, and call `diagnose_from_image`.
    - `ask_manual(question)` — wrapper to `self.chain.ask(question)` for knowledge-base queries.

- `src/data_preprocessing/generate_spectrograms.py` ([src/data_preprocessing/generate_spectrograms.py](src/data_preprocessing/generate_spectrograms.py))
  - `extract_de_signal(mat)` — extract relevant signal from `.mat`.
  - `generate_spectrogram_image(...)` — produce an image array.
  - `save_spectrogram(img_array, path)` — write a PNG for the CNN.

## Config and environment
- `configs/config.yaml` contains top-level `paths` (cnn model path, chroma db, manual_chunks, fault_cases, spectrograms) and `rag` section:
  - `rag.embedding_model` — e.g., `sentence-transformers/all-MiniLM-L6-v2`.
  - `rag.ollama_model` — default model name.
  - `rag.temperature`, `rag.top_k` — LLM and retrieval settings.

- Environment variables (via `.env`) override behavior:
  - `OLLAMA_MODEL` — model to use locally.
  - `LLM_PROVIDER` — currently set to `ollama` in the codebase.

## LLM / Ollama notes
- The project expects a local Ollama installation. Common issues:
  - Ollama 404: model not found. Fix: `ollama list` to see available models and `ollama pull <model>` to install. The code now tries to map a simple name (e.g., `llama3`) to installed variants (`llama3.2:3b`).
  - Ensure the Ollama CLI is on PATH or the Python `ollama` client is installed.
- The `get_llm()` function in `src/rag/retrieval_chain.py` performs these checks and raises clear RuntimeErrors with actionable advice.

## RAG and embeddings
- Embeddings are created using `sentence-transformers` (configurable model). The vector store is persistent under the path in `configs/config.yaml` (e.g., `data/chroma_db`).
- Rebuild the vector store when the knowledge base changes with:

```powershell
python -m src.rag.vector_store  # or call build_vector_store() from scripts
```

## Troubleshooting
- `Ollama model not found (404)`: run `ollama list` and `ollama pull <model>`; set `OLLAMA_MODEL` to installed model.
- `ModuleNotFoundError: No module named 'src'`: The app prepends `src` parent to `sys.path` in `src/app/main.py`. If you run modules from a different cwd, ensure the working directory is the repo root or run `python -m src.app.main` from repo root. Some imports were normalized to package-style (e.g., `data_preprocessing.*`) to avoid this error.
- Large files / virtual env: don't commit `.venv/` into version control. Keep only `requirements.txt` and `README.md`.

## Development notes (where to add features)
- Add new RAG prompts in `src/rag/retrieval_chain.py` near `DIAGNOSIS_PROMPT_TEMPLATE` and `QA_PROMPT_TEMPLATE`.
- To change the embedding model, edit `rag.embedding_model` in `configs/config.yaml`.
- To add new manual text chunks, update `data/manual_chunks.json` and rebuild the vector store.

## Common commands
- Run Streamlit UI:

```powershell
streamlit run src/app/main.py
```

- Rebuild vector store:

```powershell
python -c "from src.rag.vector_store import build_vector_store; build_vector_store()"
```

- Train CNN (if implementing):

```powershell
python -m src.models.train
```

## Files you may want to remove from the repository
- Temporary/duplicated files such as `arr.txt`, `Readme.txt`, local notebooks in `notebooks/`, experimental scripts like `agent.py`, and the `.venv/` folder (if present). Keep large model artifacts only if you intend to distribute them.

## Contact points in code (quick map)
- Streamlit app: [src/app/main.py](src/app/main.py)
- Pipeline entry: [src/rag/diagnosis_pipeline.py](src/rag/diagnosis_pipeline.py)
- RAG core: [src/rag/retrieval_chain.py](src/rag/retrieval_chain.py)
- Vector store: [src/rag/vector_store.py](src/rag/vector_store.py)
- CNN model/training: [src/models/cnn_classifier.py](src/models/cnn_classifier.py), [src/models/train.py](src/models/train.py)
- Data preprocessing: [src/data_preprocessing/generate_spectrograms.py](src/data_preprocessing/generate_spectrograms.py)

---

If you want, I can also:
- Add a `CONTRIBUTING.md` with development steps and code style rules.
- Add a Streamlit startup panel that lists available Ollama models and the one selected.
- Remove the candidate unnecessary files after your confirmation.

Tell me which of these you'd like next.