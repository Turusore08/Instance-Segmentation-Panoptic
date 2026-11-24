from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_OUTPUT_DIR = PROJECT_ROOT / "kaggle_output_viscom"
CKPT_DIR = KAGGLE_OUTPUT_DIR / "checkpoints"
LOG_DIR = KAGGLE_OUTPUT_DIR / "logs"
EXPORT_DIR = KAGGLE_OUTPUT_DIR / "exports_tflite"
ASSETS_DIR = PROJECT_ROOT / "assets"
PER_CLASS_AP_PATH = ASSETS_DIR / "per_class_ap.json"
BENCHMARK_CSV_PATH = KAGGLE_OUTPUT_DIR / "gpu_benchmark.csv"
TFLITE_REPORT_PATH = EXPORT_DIR / "tflite_export_report.json"

ORIG_DATASET_DIR = PROJECT_ROOT / "panoptic-indoor-segmentation.v1i.yolov8-obb"
MANUAL_DATASET_DIR = PROJECT_ROOT / "foto manual" / "outputs" / "dataset"
MANUAL_YAML_PATH = PROJECT_ROOT / "foto manual" / "data.yaml"

IMG_SIZE = 448

FALLBACK_CLASS_NAMES: Dict[int, str] = {
    0: "backpack", 1: "bed", 2: "blanket", 3: "book", 4: "bottle", 5: "bowl",
    6: "box", 7: "cabinet-merged", 8: "cardboard", 9: "ceiling-merged",
    10: "cell phone", 11: "chair", 12: "clock", 13: "couch", 14: "counter",
    15: "cup", 16: "curtain", 17: "dining table", 18: "door-stuff",
    19: "fire-extinguisher", 20: "floor-other-merged", 21: "floor-wood",
    22: "flower", 23: "food-other-merged", 24: "fork", 25: "hair drier",
    26: "handbag", 27: "keyboard", 28: "knife", 29: "laptop", 30: "light",
    31: "microwave", 32: "mirror-stuff", 33: "mouse", 34: "oven", 35: "pan",
    36: "paper", 37: "paper-merged", 38: "person", 39: "pillow", 40: "plate",
    41: "plug", 42: "potted plant", 43: "refrigerator", 44: "remote",
    45: "rug-merged", 46: "scissor", 47: "scissors", 48: "shelf", 49: "sink",
    50: "spoon", 51: "st-", 52: "stairs", 53: "stove-merged",
    54: "table-merged", 55: "teddy bear", 56: "toaster", 57: "toilet",
    58: "toothbrush", 59: "towel", 60: "toy", 61: "trash-bin",
    62: "tree-merged", 63: "tv", 64: "vase", 65: "wall", 66: "wall-brick",
    67: "wall-other-merged", 68: "wall-stone", 69: "wall-tile",
    70: "wall-wood", 71: "wallet", 72: "window-blind", 73: "window-other",
}

CANONICAL_GROUPS: List[str] = [
    "wall", "floor", "ceiling", "bed", "window", "cabinet", "person", "door",
    "table", "curtain", "chair", "shelf", "light", "box", "blind", "toilet",
    "book", "counter", "computer", "bottle", "bag", "tv", "plate", "other",
]

MANUAL_TO_CANON: Dict[str, str] = {
    "wall": "wall", "floor": "floor", "ceiling": "ceiling", "bed": "bed",
    "windowpane": "window", "cabinet": "cabinet", "wardrobe": "cabinet",
    "person": "person", "door": "door", "table": "table", "desk": "table",
    "curtain": "curtain", "chair": "chair", "shelf": "shelf", "lamp": "light",
    "box": "box", "blind": "blind", "toilet": "toilet", "book": "book",
    "countertop": "counter", "computer": "computer", "bottle": "bottle",
    "bag": "bag", "crt screen": "tv", "plate": "plate", "coffee table": "table",
    "bench": "chair", "stool": "chair", "rug": "floor",
}

ORIG_TO_CANON: Dict[str, str] = {
    "wall": "wall", "wall-brick": "wall", "wall-other-merged": "wall",
    "wall-stone": "wall", "wall-tile": "wall", "wall-wood": "wall",
    "floor-other-merged": "floor", "floor-wood": "floor", "rug-merged": "floor",
    "ceiling-merged": "ceiling", "bed": "bed", "window-other": "window",
    "cabinet-merged": "cabinet", "person": "person", "door-stuff": "door",
    "table-merged": "table", "dining table": "table", "curtain": "curtain",
    "chair": "chair", "couch": "chair", "shelf": "shelf", "light": "light",
    "box": "box", "cardboard": "box", "window-blind": "blind", "toilet": "toilet",
    "book": "book", "counter": "counter", "laptop": "computer", "keyboard": "computer",
    "bottle": "bottle", "handbag": "bag", "backpack": "bag", "tv": "tv", "plate": "plate",
}

SCENARIO_LABELS: Dict[str, str] = {
    "A": "Skenario A (Tanpa Augmentasi / Raw)",
    "B": "Skenario B (Default Framework)",
    "C": "Skenario C (Custom Albumentations Indoor)",
}

MODEL_REGISTRY: Dict[str, Dict[str, object]] = {
    "YOLOv26l-seg": {
        "family": "yolo",
        "default_scenario": "B",
        "description": "Single-stage real-time instance segmentation (yolo26l-seg, stride-4 prototype mask).",
        "checkpoints": {
            "A": CKPT_DIR / "yolo26l_seg_scenario_A_best.pt",
            "B": CKPT_DIR / "yolo26l_seg_scenario_B_best.pt",
            "C": CKPT_DIR / "yolo26l_seg_scenario_C_best.pt",
        },
        "log_csvs": {
            "A": LOG_DIR / "ultralytics" / "yolo26l_seg_scenario_A" / "results.csv",
            "B": LOG_DIR / "ultralytics" / "yolo26l_seg_scenario_B" / "results.csv",
            "C": LOG_DIR / "ultralytics" / "yolo26l_seg_scenario_C" / "results.csv",
        },
    },
    "Mask2Former-Large": {
        "family": "mask2former",
        "default_scenario": "B",
        "description": "Transformer masked-attention universal segmentation (Swin-Large backbone, 100 object queries).",
        "checkpoints": {
            "A": CKPT_DIR / "mask2former_large_scenario_A",
            "B": CKPT_DIR / "mask2former_large_scenario_B",
            "C": CKPT_DIR / "mask2former_large_scenario_C",
        },
        "log_csvs": {
            "A": LOG_DIR / "mask2former_large_scenario_A" / "results.csv",
            "B": LOG_DIR / "mask2former_large_scenario_B" / "results.csv",
            "C": LOG_DIR / "mask2former_large_scenario_C" / "results.csv",
        },
    },
    "Mask-RCNN-Large": {
        "family": "maskrcnn",
        "default_scenario": "C",
        "description": "Two-stage RoIAlign 28x28 detector & mask predictor (ResNet-101-FPN).",
        "checkpoints": {
            "A": CKPT_DIR / "maskrcnn_large_scenario_A_best.pth",
            "B": CKPT_DIR / "maskrcnn_large_scenario_B_best.pth",
            "C": CKPT_DIR / "maskrcnn_large_scenario_C_best.pth",
        },
        "log_csvs": {
            "A": LOG_DIR / "maskrcnn_large_scenario_A" / "results.csv",
            "B": LOG_DIR / "maskrcnn_large_scenario_B" / "results.csv",
            "C": LOG_DIR / "maskrcnn_large_scenario_C" / "results.csv",
        },
    },
}


def load_orig_class_names() -> Dict[int, str]:
    yaml_path = ORIG_DATASET_DIR / "data.yaml"
    if not yaml_path.exists():
        return dict(FALLBACK_CLASS_NAMES)
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    if isinstance(names, dict):
        return {int(k): str(v).strip() for k, v in names.items()}
    return {i: str(v).strip() for i, v in enumerate(names)}


def load_manual_class_names() -> Dict[int, str]:
    if not MANUAL_YAML_PATH.exists():
        return {}
    with open(MANUAL_YAML_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    if isinstance(names, dict):
        return {int(k): str(v).strip() for k, v in names.items()}
    return {i: str(v).strip() for i, v in enumerate(names)}


def get_class_color(category_id: int) -> tuple[int, int, int]:
    """Generate a deterministic, high-contrast RGB color for a class ID."""
    r = (37 * (category_id + 3) + 60) % 215 + 35
    g = (97 * (category_id + 7) + 40) % 215 + 35
    b = (151 * (category_id + 11) + 20) % 215 + 35
    return int(r), int(g), int(b)


def list_sample_images(source: str, limit: int = 40) -> List[Dict[str, str]]:
    """Return sample image paths and matching label paths for 'orig' or 'manual'."""
    samples: List[Dict[str, str]] = []
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    if source == "orig":
        img_dir = ORIG_DATASET_DIR / "test" / "images"
        lbl_dir = ORIG_DATASET_DIR / "test" / "labels"
        if img_dir.exists():
            for p in sorted(img_dir.iterdir()):
                if p.suffix.lower() in exts:
                    lf = lbl_dir / f"{p.stem}.txt"
                    samples.append({
                        "label": p.name,
                        "image_path": str(p),
                        "label_path": str(lf) if lf.exists() else "",
                        "source": "orig",
                    })
                    if len(samples) >= limit:
                        break
    elif source == "manual":
        for sp in ["test", "valid", "train"]:
            img_dir = MANUAL_DATASET_DIR / sp / "images"
            lbl_dir = MANUAL_DATASET_DIR / sp / "labels"
            if not img_dir.exists():
                continue
            for p in sorted(img_dir.iterdir()):
                if p.suffix.lower() in exts:
                    lf = lbl_dir / f"{p.stem}.txt"
                    samples.append({
                        "label": f"[{sp}] {p.name}",
                        "image_path": str(p),
                        "label_path": str(lf) if lf.exists() else "",
                        "source": "manual",
                    })
                    if len(samples) >= limit:
                        return samples
    return samples
