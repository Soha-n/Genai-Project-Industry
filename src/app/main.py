
"""
Industrial Fault Diagnosis System — Streamlit Application

Run with:  streamlit run src/app/main.py
"""

# Ensure src/ is in sys.path for imports like 'rag.*'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

CONFIG_PATH = str((Path(__file__).resolve().parent.parent / "configs/config.yaml").resolve())


@st.cache_data
def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_fault_cases():
    cfg = load_config()
    path = Path(cfg["paths"]["fault_cases"])
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return []


def get_pipeline():
    """Lazy-load the diagnosis pipeline (cached in session state)."""
    if "pipeline" not in st.session_state:
        with st.spinner("Loading models and vector store..."):
            from rag.diagnosis_pipeline import DiagnosisPipeline
            st.session_state.pipeline = DiagnosisPipeline(CONFIG_PATH)
    return st.session_state.pipeline


# ─────────────────────────────────────────────
# Page: Diagnose
# ─────────────────────────────────────────────
def page_diagnose():
    st.header("🔍 Fault Diagnosis")
    st.write("Upload a spectrogram image or raw .mat signal file to diagnose bearing faults.")

    input_type = st.radio("Input type", ["Spectrogram Image", "Raw .mat Signal"], horizontal=True)

    if input_type == "Spectrogram Image":
        uploaded = st.file_uploader("Upload spectrogram image", type=["png", "jpg", "jpeg"])
        if uploaded:
            col1, col2 = st.columns([1, 2])
            with col1:
                img = Image.open(uploaded)
                st.image(img, caption="Uploaded Spectrogram", width="stretch")

            # Save to temp file
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name

            user_query = st.text_input(
                "Optional question",
                placeholder="e.g. What corrective actions should I take?",
            )

            if st.button("🔎 Diagnose", type="primary"):
                pipeline = get_pipeline()
                with st.spinner("Running diagnosis..."):
                    result = pipeline.diagnose_from_image(tmp_path, user_query or None)

                with col2:
                    _show_classification(result["classification"])

                st.subheader("📋 Diagnosis Report")
                st.markdown(result["diagnosis"])

                with st.expander("Retrieved Context"):
                    for doc in result["retrieved_docs"]:
                        st.markdown(f"**Source:** {doc['metadata'].get('source_type', 'unknown')} | "
                                    f"**Distance:** {doc['distance']:.4f}")
                        st.text(doc["text"][:500])
                        st.divider()

                os.unlink(tmp_path)

    else:  # Raw .mat signal
        uploaded = st.file_uploader("Upload .mat file", type=["mat"])
        if uploaded:
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name

            user_query = st.text_input(
                "Optional question",
                placeholder="e.g. What is causing this vibration pattern?",
            )

            if st.button("🔎 Diagnose", type="primary"):
                pipeline = get_pipeline()
                with st.spinner("Generating spectrogram and running diagnosis..."):
                    result = pipeline.diagnose_from_signal(tmp_path, user_query or None)

                _show_classification(result["classification"])

                st.subheader("📋 Diagnosis Report")
                st.markdown(result["diagnosis"])

                with st.expander("Retrieved Context"):
                    for doc in result["retrieved_docs"]:
                        st.markdown(f"**Source:** {doc['metadata'].get('source_type', 'unknown')}")
                        st.text(doc["text"][:500])
                        st.divider()

                os.unlink(tmp_path)


def _show_classification(classification):
    """Display CNN classification results."""
    st.subheader("🤖 CNN Classification")
    pred = classification["predicted_class"]
    conf = classification["confidence"]

    st.metric("Predicted Fault", pred, f"{conf:.1%} confidence")

    # Top-3 bar chart
    top3_df = pd.DataFrame(classification["top3"])
    st.bar_chart(top3_df.set_index("class")["confidence"], width="stretch")


# ─────────────────────────────────────────────
# Page: Knowledge Base
# ─────────────────────────────────────────────
def page_knowledge_base():
    st.header("📚 Bearing Knowledge Base")
    st.write("Ask questions about rolling bearings using the maintenance handbook.")

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    question = st.chat_input("Ask about bearings...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        pipeline = get_pipeline()
        with st.chat_message("assistant"):
            with st.spinner("Searching handbook..."):
                result = pipeline.ask_manual(question)
            st.markdown(result["answer"])

            with st.expander("Sources"):
                for doc in result["retrieved_docs"]:
                    page = doc["metadata"].get("page", "?")
                    st.caption(f"Page {page} — distance: {doc['distance']:.4f}")
                    st.text(doc["text"][:300])

        st.session_state.chat_history.append({"role": "assistant", "content": result["answer"]})


# ─────────────────────────────────────────────
# Page: Case History
# ─────────────────────────────────────────────
def page_case_history():
    st.header("📁 Fault Case Database")

    cases = load_fault_cases()
    if not cases:
        st.warning("No fault cases loaded. Run `python -m src.data_preprocessing.build_fault_cases` first.")
        return

    # Filter
    severities = sorted(set(c["severity"] for c in cases))
    locations = sorted(set(c.get("location", "N/A") for c in cases))

    col1, col2 = st.columns(2)
    with col1:
        sel_severity = st.multiselect("Filter by severity", severities, default=severities)
    with col2:
        sel_location = st.multiselect("Filter by location", locations, default=locations)

    filtered = [
        c for c in cases
        if c["severity"] in sel_severity and c.get("location", "N/A") in sel_location
    ]

    for case in filtered:
        severity_color = {"none": "🟢", "minor": "🟡", "moderate": "🟠", "severe": "🔴"}.get(case["severity"], "⚪")
        with st.expander(f"{severity_color} {case['fault_type']} — {case['severity'].upper()}"):
            st.markdown(f"**Location:** {case.get('location', 'N/A')}")
            st.markdown(f"**Defect Size:** {case.get('defect_diameter_inches', 0)}\" diameter")
            st.markdown(f"**Symptoms:** {case['symptoms']}")
            st.markdown(f"**Spectrogram Pattern:** {case['spectrogram_pattern']}")
            st.markdown(f"**Root Cause:** {case['root_cause']}")
            st.markdown(f"**Recommended Action:** {case['recommended_action']}")
            st.markdown(f"**Similar Cases:** {case['similar_cases']}")


# ─────────────────────────────────────────────
# Page: Dashboard
# ─────────────────────────────────────────────
def page_dashboard():
    st.header("📊 Dashboard")
    cfg = load_config()

    # Dataset overview
    st.subheader("Dataset Overview")
    spec_dir = Path(cfg["paths"]["spectrograms"])
    if spec_dir.exists():
        class_counts = {}
        for class_dir in sorted(spec_dir.iterdir()):
            if class_dir.is_dir():
                count = len(list(class_dir.glob("*.png")))
                class_counts[class_dir.name] = count

        if class_counts:
            df = pd.DataFrame(list(class_counts.items()), columns=["Fault Class", "Samples"])
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(df, width="stretch")
                st.metric("Total Spectrograms", sum(class_counts.values()))
            with col2:
                st.bar_chart(df.set_index("Fault Class"), width="stretch")
        else:
            st.info("No spectrogram images found. Run the spectrogram generator first.")
    else:
        st.info("Spectrogram directory not found. Run the spectrogram generator first.")

    # Feature data overview
    st.subheader("Feature Dataset")
    csv_path = Path(cfg["paths"]["feature_csv"])
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        st.write(f"Shape: {df.shape[0]} samples × {df.shape[1]} features")
        st.dataframe(df.head(10), width="stretch")

        st.write("Class distribution:")
        st.bar_chart(df["fault"].value_counts())

    # Model info
    st.subheader("Model Status")
    model_path = Path(cfg["paths"]["cnn_model"])
    if model_path.exists():
        import torch
        ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
        st.success(f"CNN model loaded — {ckpt['num_classes']} classes, "
                   f"best val accuracy: {ckpt.get('val_acc', 0):.4f}, "
                   f"epoch: {ckpt.get('epoch', '?')}")
        st.write(f"Classes: {', '.join(ckpt['class_names'])}")
    else:
        st.warning("No trained CNN model found. Train the model first.")

    # Bearing test bench images
    st.subheader("Bearing Test Bench Reference Images")
    img_dir = Path(cfg["paths"]["bearing_images"])
    if img_dir.exists():
        images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")])
        if images:
            cols = st.columns(4)
            for i, img_path in enumerate(images[:8]):
                with cols[i % 4]:
                    try:
                        st.image(str(img_path), caption=img_path.stem, use_container_width=True)
                    except Exception:
                        st.caption(img_path.name)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Industrial Fault Diagnosis System",
        page_icon="⚙️",
        layout="wide",
    )

    st.sidebar.title("⚙️ Fault Diagnosis System")
    page = st.sidebar.radio(
        "Navigate",
        ["🔍 Diagnose", "📚 Knowledge Base", "📁 Case History", "📊 Dashboard"],
    )

    if page == "🔍 Diagnose":
        page_diagnose()
    elif page == "📚 Knowledge Base":
        page_knowledge_base()
    elif page == "📁 Case History":
        page_case_history()
    elif page == "📊 Dashboard":
        page_dashboard()

    st.sidebar.divider()
    st.sidebar.caption("Industrial Fault Diagnosis System v1.0")
    st.sidebar.caption("CWRU Bearing Dataset + RAG")


if __name__ == "__main__":
    main()
