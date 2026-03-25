# Industrial Fault Diagnosis System

A multimodal RAG-based fault diagnosis system for industrial bearing maintenance. Combines CNN-based vibration spectrogram classification with retrieval-augmented generation (RAG) from maintenance manuals and fault case databases.

## Features

- **Fault Diagnosis** — Upload a spectrogram image or raw .mat signal to get automated fault classification with corrective action recommendations
- **Knowledge Base Q&A** — Chat with the rolling bearing handbook using RAG
- **Fault Case Database** — Browse structured fault cases with symptoms, root causes, and recommended actions
- **Dashboard** — Visualize dataset statistics and model performance

## Architecture

```
Raw .mat Signal → Spectrogram Generation → CNN Classification (ResNet-18)
                                                    ↓
                                              Fault Class + Confidence
                                                    ↓
                                    RAG Retrieval (ChromaDB + Manual + Cases)
                                                    ↓
                                          LLM Diagnosis Report
                                                    ↓
                                       Streamlit Web Interface
```

## Project Structure

```
├── configs/
│   └── config.yaml                 # Centralized configuration
├── data/
│   ├── CWRU Bearing Dataset/
│   │   ├── raw/                    # Raw .mat vibration signals
│   │   ├── CWRU_48k_load_1_CNN_data.npz
│   │   └── feature_time_48k_2048_load_1.csv
│   ├── bearing-images/             # Test bench reference images
│   ├── rolling-bearing-handbook.pdf
│   ├── spectrograms/               # Generated spectrogram images (per class)
│   ├── manual_chunks.json          # Extracted PDF text chunks
│   ├── fault_cases.json            # Structured fault case KB
│   └── chroma_db/                  # Persisted vector store
├── models/
│   └── cnn_bearing_fault.pth       # Trained CNN weights
├── notebooks/
│   └── 01_data_exploration.ipynb   # Data exploration & visualization
├── src/
│   ├── data_preprocessing/
│   │   ├── generate_spectrograms.py
│   │   ├── extract_pdf.py
│   │   └── build_fault_cases.py
│   ├── models/
│   │   ├── cnn_classifier.py       # ResNet-18 architecture
│   │   ├── train.py                # CNN training loop
│   │   └── feature_classifier.py   # sklearn baseline
│   ├── rag/
│   │   ├── vector_store.py         # ChromaDB embeddings
│   │   ├── retrieval_chain.py      # LangChain QA chain
│   │   └── diagnosis_pipeline.py   # Multimodal pipeline
│   └── app/
│       └── main.py                 # Streamlit UI
├── requirements.txt
└── .env.example
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key (or configure Ollama for local LLM)
```

### 3. Data Preprocessing

Run these commands from the project root:

```bash
# Generate spectrogram images from raw .mat signals
python -m src.data_preprocessing.generate_spectrograms

# Extract text from the bearing handbook PDF
python -m src.data_preprocessing.extract_pdf

# Build the fault case knowledge base
python -m src.data_preprocessing.build_fault_cases
```

### 4. Build Vector Store

```bash
python -m src.rag.vector_store
```

### 5. Train the CNN Model

```bash
python -m src.models.train
```

### 6. (Optional) Run Feature Baseline

```bash
python -m src.models.feature_classifier
```

### 7. Launch the App

```bash
streamlit run src/app/main.py
```

## Dataset

**CWRU Bearing Dataset** — Case Western Reserve University bearing vibration data. 10 fault conditions:

| Class | Location | Defect Size |
|-------|----------|-------------|
| Normal | N/A | 0" |
| Ball_007 | Rolling element | 0.007" |
| Ball_014 | Rolling element | 0.014" |
| Ball_021 | Rolling element | 0.021" |
| IR_007 | Inner race | 0.007" |
| IR_014 | Inner race | 0.014" |
| IR_021 | Inner race | 0.021" |
| OR_007 | Outer race | 0.007" |
| OR_014 | Outer race | 0.014" |
| OR_021 | Outer race | 0.021" |

## Technology Stack

- **Deep Learning**: PyTorch (ResNet-18 transfer learning)
- **Vector Database**: ChromaDB
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: OpenAI GPT-3.5/4 or Ollama (local)
- **RAG Framework**: LangChain
- **Signal Processing**: librosa, scipy
- **UI**: Streamlit
