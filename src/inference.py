from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from ultralytics import YOLO

from src.config import IMG_SIZE, get_class_color, load_orig_class_names


@dataclass
class Detection:
    category_id: int
    class_name: str
    score: float
    mask: np.ndarray  # uint8 binary (H, W)
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    area_px: int
    area_pct: float


@dataclass
class GroundTruthObject:
    category_id: int
    class_name: str
    mask: np.ndarray  # uint8 binary (H, W)
    bbox: Tuple[int, int, int, int]
    area_px: int


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def preprocess_domain_alignment(img_rgb: np.ndarray, apply_grayscale_clahe: bool) -> np.ndarray:
    """Optionally align RGB photos to the Grayscale CRT contrast distribution of the training set."""
    if not apply_grayscale_clahe:
        return img_rgb
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)


def _mask_to_bbox(bin_mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(bin_mask > 0)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def load_yolo_model(ckpt_path: str | Path) -> YOLO:
    return YOLO(str(ckpt_path))


def load_mask2former_model(ckpt_dir: str | Path) -> Tuple[Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation]:
    processor = Mask2FormerImageProcessor.from_pretrained(str(ckpt_dir))
    model = Mask2FormerForUniversalSegmentation.from_pretrained(str(ckpt_dir))
    model.to(get_device())
    model.eval()
    return processor, model


def load_maskrcnn_model(ckpt_path: str | Path, num_classes: int = 74) -> MaskRCNN:
    backbone = resnet_fpn_backbone(backbone_name="resnet101", weights=None, trainable_layers=5)
    model = MaskRCNN(backbone, num_classes=num_classes + 1, min_size=IMG_SIZE, max_size=IMG_SIZE)
    in_b = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_b, num_classes + 1)
    in_m = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_m, 256, num_classes + 1)
    state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state)
    model.to(get_device())
    model.eval()
    return model


def predict_yolo(
    model: YOLO,
    img_rgb: np.ndarray,
    conf_thresh: float = 0.05,
    iou_thresh: float = 0.70,
    imgsz: int = 448,
    class_names: Optional[Dict[int, str]] = None,
) -> Tuple[List[Detection], float]:
    class_names = class_names or load_orig_class_names()
    h, w = img_rgb.shape[:2]
    total_px = max(h * w, 1)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    t0 = time.perf_counter()
    res = model.predict(
        img_bgr,
        imgsz=imgsz,
        conf=conf_thresh,
        iou=iou_thresh,
        max_det=300,
        verbose=False,
        device=get_device(),
    )[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    detections: List[Detection] = []
    if res.masks is not None and res.boxes is not None:
        masks = res.masks.data.cpu().numpy()
        boxes = res.boxes
        for i in range(len(boxes)):
            m = masks[i]
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            bm = (m > 0.5).astype(np.uint8)
            area_px = int(bm.sum())
            if area_px == 0:
                continue
            cid = int(boxes.cls[i].item())
            detections.append(
                Detection(
                    category_id=cid,
                    class_name=class_names.get(cid, f"cls_{cid}"),
                    score=float(boxes.conf[i].item()),
                    mask=bm,
                    bbox=_mask_to_bbox(bm),
                    area_px=area_px,
                    area_pct=100.0 * area_px / total_px,
                )
            )
    detections.sort(key=lambda d: d.score, reverse=True)
    return detections, elapsed_ms


@torch.no_grad()
def predict_mask2former(
    proc_and_model: Tuple[Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation],
    img_rgb: np.ndarray,
    conf_thresh: float = 0.05,
    class_names: Optional[Dict[int, str]] = None,
) -> Tuple[List[Detection], float]:
    processor, model = proc_and_model
    class_names = class_names or load_orig_class_names()
    h, w = img_rgb.shape[:2]
    total_px = max(h * w, 1)
    device = get_device()
    resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

    t0 = time.perf_counter()
    enc = processor(images=resized, return_tensors="pt")
    pixel_values = enc["pixel_values"].to(device)
    pixel_mask = enc.get("pixel_mask")
    if pixel_mask is not None:
        pixel_mask = pixel_mask.to(device)
    outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    cls_logits = outputs.class_queries_logits
    mask_logits = outputs.masks_queries_logits

    probs = cls_logits.softmax(-1)[..., :-1]
    num_q, num_c = probs.shape[1], probs.shape[2]
    flat_scores, topk_idx = probs[0].reshape(-1).topk(min(100, num_q * num_c))
    q_idx = torch.div(topk_idx, num_c, rounding_mode="floor")
    c_idx = topk_idx % num_c
    keep = flat_scores >= conf_thresh
    detections: List[Detection] = []
    if torch.any(keep):
        q_idx, c_idx, cls_scores = q_idx[keep], c_idx[keep], flat_scores[keep]
        masks_k = mask_logits[0, q_idx]
        masks_resized = F.interpolate(masks_k.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
        mask_probs = masks_resized.sigmoid()
        bin_t = (mask_probs > 0.5).float()
        maskness = (mask_probs * bin_t).flatten(1).sum(1) / (bin_t.flatten(1).sum(1) + 1e-6)
        final_scores = (cls_scores * maskness).cpu().numpy()
        bin_masks = bin_t.cpu().numpy().astype(np.uint8)
        labels_np = c_idx.cpu().numpy()
        for sc, lbl, bm in zip(final_scores, labels_np, bin_masks):
            area_px = int(bm.sum())
            if float(sc) < conf_thresh or area_px == 0:
                continue
            cid = int(lbl)
            detections.append(
                Detection(
                    category_id=cid,
                    class_name=class_names.get(cid, f"cls_{cid}"),
                    score=float(sc),
                    mask=bm,
                    bbox=_mask_to_bbox(bm),
                    area_px=area_px,
                    area_pct=100.0 * area_px / total_px,
                )
            )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    detections.sort(key=lambda d: d.score, reverse=True)
    return detections, elapsed_ms


@torch.no_grad()
def predict_maskrcnn(
    model: MaskRCNN,
    img_rgb: np.ndarray,
    conf_thresh: float = 0.05,
    class_names: Optional[Dict[int, str]] = None,
) -> Tuple[List[Detection], float]:
    class_names = class_names or load_orig_class_names()
    num_classes = len(class_names)
    h, w = img_rgb.shape[:2]
    total_px = max(h * w, 1)
    device = get_device()
    resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    tensor_in = (torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0).to(device)

    t0 = time.perf_counter()
    res = model([tensor_in])[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    detections: List[Detection] = []
    scores = res["scores"].cpu().numpy()
    labels = res["labels"].cpu().numpy()
    masks = res["masks"].cpu().numpy()
    for sc, lbl, m in zip(scores, labels, masks):
        if float(sc) < conf_thresh:
            continue
        bm = (m[0] > 0.5).astype(np.uint8)
        if bm.shape != (h, w):
            bm = cv2.resize(bm, (w, h), interpolation=cv2.INTER_NEAREST)
        cid = int(lbl) - 1
        area_px = int(bm.sum())
        if area_px == 0 or cid < 0 or cid >= num_classes:
            continue
        detections.append(
            Detection(
                category_id=cid,
                class_name=class_names.get(cid, f"cls_{cid}"),
                score=float(sc),
                mask=bm,
                bbox=_mask_to_bbox(bm),
                area_px=area_px,
                area_pct=100.0 * area_px / total_px,
            )
        )
    detections.sort(key=lambda d: d.score, reverse=True)
    return detections, elapsed_ms


def parse_yolo_seg_label(
    label_text: str,
    img_w: int,
    img_h: int,
    class_names: Dict[int, str],
) -> List[GroundTruthObject]:
    """Parse YOLO segmentation/OBB polygon labels into binary masks."""
    gts: List[GroundTruthObject] = []
    for line in label_text.splitlines():
        parts = line.strip().split()
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            continue
        cid = int(float(parts[0]))
        coords = np.clip(np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2), 0.0, 1.0)
        coords[:, 0] *= img_w
        coords[:, 1] *= img_h
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(coords).astype(np.int32)], 1)
        area_px = int(mask.sum())
        if area_px <= 4:
            continue
        gts.append(
            GroundTruthObject(
                category_id=cid,
                class_name=class_names.get(cid, f"cls_{cid}"),
                mask=mask,
                bbox=_mask_to_bbox(mask),
                area_px=area_px,
            )
        )
    return gts


def render_overlay(
    img_rgb: np.ndarray,
    detections: Sequence[Detection],
    alpha: float = 0.45,
    show_boxes: bool = True,
    show_contours: bool = True,
    show_labels: bool = True,
    allowed_classes: Optional[Sequence[str]] = None,
) -> np.ndarray:
    vis = img_rgb.copy()
    allowed_set = set(allowed_classes) if allowed_classes is not None else None
    font_scale = max(0.45, min(img_rgb.shape[0], img_rgb.shape[1]) / 1100.0)
    thickness = 2 if max(img_rgb.shape[:2]) >= 800 else 1

    for det in detections:
        if allowed_set is not None and det.class_name not in allowed_set:
            continue
        col = np.array(get_class_color(det.category_id), dtype=np.float32)
        m = det.mask > 0
        vis[m] = np.clip(vis[m].astype(np.float32) * (1.0 - alpha) + col * alpha, 0, 255).astype(np.uint8)
        col_tuple = tuple(int(c) for c in col)
        if show_contours:
            cnts, _ = cv2.findContours(det.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, col_tuple, thickness + 1)
        x1, y1, x2, y2 = det.bbox
        if show_boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), col_tuple, thickness)
        if show_labels:
            txt = f"{det.class_name} {det.score:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            ty = max(th + 6, y1)
            cv2.rectangle(vis, (x1, ty - th - 6), (min(vis.shape[1] - 1, x1 + tw + 6), ty), col_tuple, -1)
            cv2.putText(vis, txt, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return vis


def render_gt_overlay(
    img_rgb: np.ndarray,
    gts: Sequence[GroundTruthObject],
    alpha: float = 0.40,
    allowed_classes: Optional[Sequence[str]] = None,
) -> np.ndarray:
    vis = img_rgb.copy()
    allowed_set = set(allowed_classes) if allowed_classes is not None else None
    font_scale = max(0.45, min(img_rgb.shape[0], img_rgb.shape[1]) / 1100.0)
    thickness = 2 if max(img_rgb.shape[:2]) >= 800 else 1

    for gt in gts:
        if allowed_set is not None and gt.class_name not in allowed_set:
            continue
        col = np.array(get_class_color(gt.category_id), dtype=np.float32)
        m = gt.mask > 0
        vis[m] = np.clip(vis[m].astype(np.float32) * (1.0 - alpha) + col * alpha, 0, 255).astype(np.uint8)
        col_tuple = tuple(int(c) for c in col)
        cnts, _ = cv2.findContours(gt.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, col_tuple, thickness + 1)
        x1, y1, _, _ = gt.bbox
        (tw, th), _ = cv2.getTextSize(gt.class_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        ty = max(th + 6, y1)
        cv2.rectangle(vis, (x1, ty - th - 6), (min(vis.shape[1] - 1, x1 + tw + 6), ty), col_tuple, -1)
        cv2.putText(vis, gt.class_name, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return vis


def render_single_class_isolation(
    img_rgb: np.ndarray,
    detections: Sequence[Detection],
    target_class: str,
    alpha: float = 0.60,
    gts: Optional[Sequence[GroundTruthObject]] = None,
    use_canonical: bool = False,
    canon_map_pred: Optional[Dict[str, str]] = None,
    canon_map_gt: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    """Dim background outside target_class predictions and highlight GT outlines if available."""
    dimmed = (img_rgb.astype(np.float32) * 0.35).astype(np.uint8)
    vis = dimmed.copy()
    matched_preds = [
        d for d in detections
        if (canon_map_pred.get(d.class_name, d.class_name) if use_canonical and canon_map_pred else d.class_name) == target_class
    ]
    for det in matched_preds:
        m = det.mask > 0
        col = np.array((0, 220, 140), dtype=np.float32)
        vis[m] = np.clip(img_rgb[m].astype(np.float32) * (1.0 - alpha) + col * alpha, 0, 255).astype(np.uint8)
        cnts, _ = cv2.findContours(det.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 160), 2)
        x1, y1, _, _ = det.bbox
        cv2.putText(vis, f"{det.class_name} {det.score:.2f}", (x1, max(16, y1 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    if gts:
        matched_gts = [
            g for g in gts
            if (canon_map_gt.get(g.class_name, g.class_name) if use_canonical and canon_map_gt else g.class_name) == target_class
        ]
        for gt in matched_gts:
            cnts, _ = cv2.findContours(gt.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (255, 80, 80), 2)
    return vis
