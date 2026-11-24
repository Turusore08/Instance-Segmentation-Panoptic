from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.config import (
    BENCHMARK_CSV_PATH,
    MANUAL_TO_CANON,
    MODEL_REGISTRY,
    ORIG_TO_CANON,
    PER_CLASS_AP_PATH,
    TFLITE_REPORT_PATH,
)
from src.inference import Detection, GroundTruthObject


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()
    return float(inter / union) if union > 0 else 0.0


def compute_single_image_class_metrics(
    detections: Sequence[Detection],
    gts: Sequence[GroundTruthObject],
    target_class: str,
    iou_thresh: float = 0.50,
    use_canonical: bool = False,
) -> Dict[str, Any]:
    """Compute COCO-style 101-pt AP, Precision, Recall, F1, and Mean IoU for one class on a single image."""
    if use_canonical:
        preds = [d for d in detections if ORIG_TO_CANON.get(d.class_name, d.class_name) == target_class]
        gt_items = [g for g in gts if MANUAL_TO_CANON.get(g.class_name, g.class_name) == target_class]
    else:
        preds = [d for d in detections if d.class_name == target_class]
        gt_items = [g for g in gts if g.class_name == target_class]

    preds = sorted(preds, key=lambda d: d.score, reverse=True)
    n_gt = len(gt_items)
    n_pred = len(preds)

    if n_gt == 0 and n_pred == 0:
        return {"ap": None, "precision": 0.0, "recall": 0.0, "f1": 0.0, "mean_iou": 0.0, "tp": 0, "fp": 0, "fn": 0, "n_gt": 0, "n_pred": 0}
    if n_gt == 0:
        return {"ap": None, "precision": 0.0, "recall": 0.0, "f1": 0.0, "mean_iou": 0.0, "tp": 0, "fp": n_pred, "fn": 0, "n_gt": 0, "n_pred": n_pred}
    if n_pred == 0:
        return {"ap": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "mean_iou": 0.0, "tp": 0, "fp": 0, "fn": n_gt, "n_gt": n_gt, "n_pred": 0}

    matched_gt = set()
    tp_arr = np.zeros(n_pred, dtype=np.float32)
    fp_arr = np.zeros(n_pred, dtype=np.float32)

    for i, pred in enumerate(preds):
        best_iou = 0.0
        best_gt_idx = -1
        for j, gt in enumerate(gt_items):
            if j in matched_gt:
                continue
            iou = mask_iou(pred.mask, gt.mask)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        if best_iou >= iou_thresh and best_gt_idx >= 0:
            tp_arr[i] = 1.0
            matched_gt.add(best_gt_idx)
        else:
            fp_arr[i] = 1.0

    tp_cum = np.cumsum(tp_arr)
    fp_cum = np.cumsum(fp_arr)
    recalls = tp_cum / n_gt
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    # COCO 101-point interpolation
    rec_thresholds = np.linspace(0.0, 1.0, 101)
    ap_vals = []
    for r_t in rec_thresholds:
        p_valid = precisions[recalls >= r_t]
        ap_vals.append(float(np.max(p_valid)) if len(p_valid) > 0 else 0.0)
    ap = float(np.mean(ap_vals))

    tp_total = int(tp_arr.sum())
    fp_total = int(fp_arr.sum())
    fn_total = n_gt - tp_total
    prec = tp_total / n_pred if n_pred > 0 else 0.0
    rec = tp_total / n_gt if n_gt > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    gt_best_ious = [max((mask_iou(g.mask, p.mask) for p in preds), default=0.0) for g in gt_items]

    return {
        "ap": round(ap, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "mean_iou": round(float(np.mean(gt_best_ious)), 4),
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "n_gt": n_gt,
        "n_pred": n_pred,
    }


def load_per_class_ap_data() -> Optional[Dict[str, Any]]:
    if not PER_CLASS_AP_PATH.exists():
        return None
    with open(PER_CLASS_AP_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_gpu_benchmark() -> pd.DataFrame:
    if not BENCHMARK_CSV_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(BENCHMARK_CSV_PATH)


def load_training_log(model_name: str, scenario: str) -> pd.DataFrame:
    spec = MODEL_REGISTRY.get(model_name, {})
    csv_map = spec.get("log_csvs", {})
    csv_path: Optional[Path] = csv_map.get(scenario) if isinstance(csv_map, dict) else None
    if csv_path is None or not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_tflite_export_report() -> List[Dict[str, Any]]:
    if not TFLITE_REPORT_PATH.exists():
        return []
    with open(TFLITE_REPORT_PATH, encoding="utf-8") as f:
        return json.load(f)
