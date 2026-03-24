"""
Generate spectrogram images from CWRU Bearing Dataset .mat files.

Reads raw vibration signals, segments them into fixed-length windows,
and produces mel-spectrogram images saved per fault class.
"""

import os
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.io import loadmat
from pathlib import Path
from tqdm import tqdm


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# CWRU .mat files use different key names for the DE accelerometer channel.
# The key always contains "DE_time" as a substring.
def extract_de_signal(mat_data):
    """Extract the Drive-End accelerometer signal from a .mat dict."""
    for key in mat_data:
        if "DE_time" in key:
            return mat_data[key].flatten()
    raise KeyError(f"No DE_time key found. Available keys: {list(mat_data.keys())}")


def segment_signal(signal, segment_length, hop=None):
    """Split signal into overlapping segments."""
    if hop is None:
        hop = segment_length // 2
    segments = []
    for start in range(0, len(signal) - segment_length + 1, hop):
        segments.append(signal[start : start + segment_length])
    return segments


def generate_spectrogram_image(segment, sr, n_fft, hop_length, n_mels, image_size):
    """Generate a mel-spectrogram image from a signal segment.
    Returns the image as a numpy uint8 RGB array.
    """
    S = librosa.feature.melspectrogram(
        y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(1, 1, figsize=(2.24, 2.24), dpi=100)
    librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, ax=ax)
    ax.axis("off")
    plt.tight_layout(pad=0)

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    # Resize to exact target size
    from PIL import Image
    img_pil = Image.fromarray(img).resize((image_size, image_size), Image.LANCZOS)
    return np.array(img_pil)


def save_spectrogram(img_array, save_path):
    from PIL import Image
    img = Image.fromarray(img_array)
    img.save(save_path)


def run(config_path="configs/config.yaml"):
    cfg = load_config(config_path)
    raw_dir = Path(cfg["paths"]["raw_data"])
    out_dir = Path(cfg["paths"]["spectrograms"])
    spec_cfg = cfg["spectrogram"]
    mat_mapping = cfg["mat_file_mapping"]

    sr = spec_cfg["sample_rate"]
    seg_len = spec_cfg["segment_length"]
    hop_length = spec_cfg["hop_length"]
    n_mels = spec_cfg["n_mels"]
    n_fft = spec_cfg["n_fft"]
    image_size = spec_cfg["image_size"]

    total_images = 0

    for mat_file in sorted(raw_dir.glob("*.mat")):
        # Determine fault class from filename
        fault_class = None
        for prefix, cls in mat_mapping.items():
            if mat_file.stem.startswith(prefix):
                fault_class = cls
                break
        if fault_class is None:
            print(f"  [SKIP] Unknown mapping for {mat_file.name}")
            continue

        class_dir = out_dir / fault_class
        class_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {mat_file.name} → {fault_class}")
        mat_data = loadmat(str(mat_file))
        signal = extract_de_signal(mat_data)

        segments = segment_signal(signal, seg_len)
        print(f"  {len(segments)} segments from {len(signal)} samples")

        for i, seg in enumerate(tqdm(segments, desc=f"  {fault_class}", leave=False)):
            img = generate_spectrogram_image(seg, sr, n_fft, hop_length, n_mels, image_size)
            fname = f"{fault_class}_{mat_file.stem}_{i:04d}.png"
            save_spectrogram(img, class_dir / fname)
            total_images += 1

    print(f"\nDone — generated {total_images} spectrogram images in {out_dir}")


if __name__ == "__main__":
    run()
