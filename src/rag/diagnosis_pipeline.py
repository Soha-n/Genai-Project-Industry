"""
Multimodal diagnosis pipeline that combines:
  1. CNN-based spectrogram classification
  2. RAG-based knowledge retrieval and LLM generation

Input: spectrogram image (or raw .mat signal)
Output: structured diagnosis report
"""

import yaml
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

from src.models.cnn_classifier import BearingFaultCNN
from src.rag.vector_store import get_retriever
from src.rag.retrieval_chain import RetrievalChain, get_llm


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class DiagnosisPipeline:
    """End-to-end bearing fault diagnosis pipeline."""

    def __init__(self, config_path="configs/config.yaml"):
        self.cfg = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_cnn()
        self._load_rag()

    def _load_cnn(self):
        """Load the trained CNN model."""
        model_path = Path(self.cfg["paths"]["cnn_model"])
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {model_path}. "
                "Run 'python -m src.models.train' first."
            )

        checkpoint = torch.load(str(model_path), map_location=self.device, weights_only=False)
        self.class_names = checkpoint["class_names"]
        num_classes = checkpoint["num_classes"]

        self.model = BearingFaultCNN(num_classes=num_classes, pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        image_size = self.cfg["spectrogram"]["image_size"]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _load_rag(self):
        """Initialize the RAG retrieval chain."""
        retriever = get_retriever(self.cfg)
        llm = get_llm(self.cfg)
        self.chain = RetrievalChain(retriever, llm)

    def classify_image(self, image_path):
        """Classify a spectrogram image and return predictions."""
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        pred, probs = self.model.predict(tensor)
        pred_class = self.class_names[pred.item()]
        confidence = probs[0][pred.item()].item()

        # Top-3 predictions
        top3_probs, top3_idx = probs[0].topk(3)
        top3 = [
            {"class": self.class_names[idx.item()], "confidence": prob.item()}
            for prob, idx in zip(top3_probs, top3_idx)
        ]

        return {
            "predicted_class": pred_class,
            "confidence": confidence,
            "top3": top3,
            "all_probs": {self.class_names[i]: probs[0][i].item() for i in range(len(self.class_names))},
        }

    def diagnose_from_image(self, image_path, user_query=None):
        """Full diagnosis pipeline: classify image → RAG retrieval → LLM report."""
        # Step 1: CNN classification
        classification = self.classify_image(image_path)

        # Step 2: Build diagnosis query
        fault_class = classification["predicted_class"]
        confidence = classification["confidence"]
        query = user_query or f"Diagnose the detected {fault_class} bearing fault and recommend corrective actions."

        # Step 3: RAG retrieval + LLM generation
        rag_result = self.chain.diagnose(query, fault_class=fault_class, confidence=confidence)

        return {
            "classification": classification,
            "diagnosis": rag_result["answer"],
            "retrieved_docs": rag_result["retrieved_docs"],
            "query": rag_result["query"],
        }

    def diagnose_from_signal(self, mat_file_path, user_query=None):
        """Diagnose from a raw .mat file: generate spectrogram → classify → RAG."""
        from src.data_preprocessing.generate_spectrograms import (
            extract_de_signal,
            generate_spectrogram_image,
            save_spectrogram,
        )
        from scipy.io import loadmat
        import tempfile

        scfg = self.cfg["spectrogram"]
        mat_data = loadmat(str(mat_file_path))
        signal = extract_de_signal(mat_data)

        # Take first segment
        seg_len = scfg["segment_length"]
        segment = signal[:seg_len]

        img_array = generate_spectrogram_image(
            segment, scfg["sample_rate"], scfg["n_fft"],
            scfg["hop_length"], scfg["n_mels"], scfg["image_size"]
        )

        # Save to temp file and classify
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            save_spectrogram(img_array, tmp.name)
            tmp_path = tmp.name

        try:
            result = self.diagnose_from_image(tmp_path, user_query)
            result["source"] = "raw_signal"
            result["mat_file"] = str(mat_file_path)
            return result
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def ask_manual(self, question):
        """Query the bearing manual knowledge base directly."""
        return self.chain.ask(question)
