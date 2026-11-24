from __future__ import annotations

import gc
import json
import math
import os
import time
import warnings
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
import numpy as np
import pandas as pd
import pillow_heif
pillow_heif.register_heif_opener()
import torch
from PIL import Image, ImageOps
from transformers import (
    AutoImageProcessor,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Mask2FormerForUniversalSegmentation,
)
from ultralytics import SAM

warnings.filterwarnings("ignore")
torch.set_num_threads(min(16, os.cpu_count() or 8))

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = Path("/home/turusore08/Desktop/Semester 5/Computer Vision/day 10 Activity")
PHOTOS_DIR = BASE_DIR / "viskom-data-sendiri"
OUTPUT_DIR = BASE_DIR / "sota_auto_annotation_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "visualizations").mkdir(exist_ok=True)

WEIGHTS_CACHE = Path("/tmp/sota_weights")
WEIGHTS_CACHE.mkdir(parents=True, exist_ok=True)
SAM2_WEIGHTS = str(WEIGHTS_CACHE / "sam2.1_l.pt")

MASK2FORMER_MODEL_ID = "facebook/mask2former-swin-large-coco-panoptic"
GDINO_MODEL_ID = "IDEA-Research/grounding-dino-base"

M2F_THRESHOLD = 0.45
GDINO_BOX_THRESHOLD = 0.28
GDINO_TEXT_THRESHOLD = 0.24
MIN_MASK_AREA_RATIO = 0.00015
MERGE_IOU_THRESHOLD = 0.55
POLY_EPSILON_RATIO = 0.0015
MAX_INFER_LONG_EDGE = 1280
INCLUDE_ROBOFLOW_ROOT_CAT = False

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CLASS_NAMES_74 = [
    "backpack", "bed", "blanket", "book", "bottle", "bowl", "box",
    "cabinet-merged", "cardboard", "ceiling-merged", "cell phone", "chair",
    "clock", "couch", "counter", "cup", "curtain", "dining table",
    "door-stuff", "fire-extinguisher", "floor-other-merged", "floor-wood",
    "flower", "food-other-merged", "fork", "hair drier", "handbag",
    "keyboard", "knife", "laptop", "light", "microwave", "mirror-stuff",
    "mouse", "oven", "pan", "paper", "paper-merged", "person", "pillow",
    "plate", "plug", "potted plant", "refrigerator", "remote", "rug-merged",
    "scissor", "scissors", "shelf", "sink", "spoon", "st-", "stairs",
    "stove-merged", "table-merged", "teddy bear", "toaster", "toilet",
    "toothbrush", "towel", "toy", "trash-bin", "tree-merged", "tv", "vase",
    "wall", "wall-brick", "wall-other-merged", "wall-stone", "wall-tile",
    "wall-wood", "wallet", "window-blind", "window-other",
]
assert len(CLASS_NAMES_74) == 74

NAME_TO_CAT_ID = {name: idx + 1 for idx, name in enumerate(CLASS_NAMES_74)}
STUFF_CLASSES = {
    "blanket", "cabinet-merged", "cardboard", "ceiling-merged", "counter",
    "curtain", "door-stuff", "floor-other-merged", "floor-wood", "flower",
    "food-other-merged", "light", "mirror-stuff", "paper-merged", "pillow",
    "rug-merged", "shelf", "stairs", "stove-merged", "table-merged", "towel",
    "tree-merged", "wall", "wall-brick", "wall-other-merged", "wall-stone",
    "wall-tile", "wall-wood", "window-blind", "window-other", "st-",
}

COCO_PANOPTIC_TO_TARGET = {name: name for name in CLASS_NAMES_74}
COCO_PANOPTIC_TO_TARGET.update({
    "wall-concrete": "wall",
    "wall-panel": "wall",
    "floor-tile": "floor-other-merged",
    "floor-marble": "floor-other-merged",
    "floor-stone": "floor-other-merged",
    "floor-carpet": "rug-merged",
    "carpet": "rug-merged",
    "rug": "rug-merged",
    "cupboard": "cabinet-merged",
    "desk-stuff": "table-merged",
    "table": "table-merged",
    "window": "window-other",
    "door": "door-stuff",
    "mirror": "mirror-stuff",
    "ceiling": "ceiling-merged",
    "cabinet": "cabinet-merged",
    "paper": "paper-merged",
})

GDINO_PROMPT_BATCHES = [
    {
        "fire extinguisher": "fire-extinguisher",
        "frying pan": "pan",
        "cooking pan": "pan",
        "plate": "plate",
        "electrical plug": "plug",
        "power outlet": "plug",
        "trash bin": "trash-bin",
        "garbage can": "trash-bin",
        "wallet": "wallet",
    },
    {
        "book": "book",
        "ceiling light": "light",
        "lamp": "light",
        "light fixture": "light",
        "spoon": "spoon",
        "cardboard box": "box",
        "storage box": "box",
        "toy": "toy",
        "stove": "stove-merged",
    },
    {
        "fork": "fork",
        "knife": "knife",
        "cup": "cup",
        "mug": "cup",
        "bottle": "bottle",
        "bowl": "bowl",
        "scissors": "scissors",
        "toothbrush": "toothbrush",
        "paper sheet": "paper",
    },
    {
        "smartphone": "cell phone",
        "remote control": "remote",
        "computer mouse": "mouse",
        "keyboard": "keyboard",
        "laptop": "laptop",
        "wall clock": "clock",
        "vase": "vase",
        "flower": "flower",
        "potted plant": "potted plant",
    },
    {
        "backpack": "backpack",
        "handbag": "handbag",
        "pillow": "pillow",
        "blanket": "blanket",
        "towel": "towel",
        "teddy bear": "teddy bear",
        "hair dryer": "hair drier",
        "chair": "chair",
        "tv monitor": "tv",
    },
]

SAM2_PRIORITY_CLASSES = {
    "book", "light", "spoon", "fork", "knife", "cup", "bottle", "bowl",
    "cell phone", "remote", "mouse", "keyboard", "clock", "vase", "scissors",
    "toothbrush", "plate", "pan", "plug", "trash-bin", "wallet", "box",
    "fire-extinguisher", "toy", "paper", "flower",
}


def build_coco_categories(include_root: bool = False) -> list[dict]:
    cats = []
    if include_root:
        cats.append({"id": 0, "name": "objects", "supercategory": "none"})
    for idx, name in enumerate(CLASS_NAMES_74, start=1):
        cats.append({
            "id": idx,
            "name": name,
            "supercategory": "stuff" if name in STUFF_CLASSES else "thing",
        })
    return cats


def load_image_exif_corrected(img_path: Path) -> Image.Image:
    with Image.open(img_path) as im:
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


def resize_for_inference(pil_img: Image.Image, max_long_edge: int = MAX_INFER_LONG_EDGE):
    w, h = pil_img.size
    long_edge = max(w, h)
    if long_edge <= max_long_edge:
        return pil_img, 1.0
    scale = max_long_edge / float(long_edge)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return pil_img.resize((new_w, new_h), Image.BILINEAR), scale


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / float(union + 1e-6)


def mask_to_polygons_and_bbox(binary_mask: np.ndarray, epsilon_ratio: float = POLY_EPSILON_RATIO):
    mask_u8 = (binary_mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 16.0:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon_ratio * peri, True)
        if len(approx) < 3:
            approx = cnt
        if len(approx) >= 3:
            poly = approx.reshape(-1).astype(float).tolist()
            if len(poly) >= 6:
                polygons.append(poly)
    if not polygons:
        return [], None, 0.0

    ys, xs = np.where(binary_mask)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    bbox = [x_min, y_min, max(1.0, x_max - x_min + 1.0), max(1.0, y_max - y_min + 1.0)]
    area = float(binary_mask.sum())
    return polygons, bbox, area


def fuse_stage_predictions(preds: list, iou_thresh: float = MERGE_IOU_THRESHOLD) -> list:
    by_class = defaultdict(list)
    for p in preds:
        by_class[p["label"]].append(p)

    fused = []
    for label, items in by_class.items():
        prefer_sam2 = label in SAM2_PRIORITY_CLASSES
        items.sort(
            key=lambda x: (
                1 if (prefer_sam2 and x["source"] == "gdino_sam2") else (1 if (not prefer_sam2 and x["source"] == "mask2former") else 0),
                x["score"],
            ),
            reverse=True,
        )
        kept = []
        for cand in items:
            duplicate = False
            for k in kept:
                if mask_iou(cand["mask"], k["mask"]) >= iou_thresh:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(cand)
        fused.extend(kept)
    return fused


def match_phrase_to_target(raw_phrase: str, batch_map: dict):
    p = raw_phrase.strip().lower()
    if not p:
        return None
    if p in batch_map:
        return batch_map[p]
    for phrase in sorted(batch_map.keys(), key=len, reverse=True):
        if phrase in p or p in phrase:
            return batch_map[phrase]
    return None


def main() -> None:
    valid_exts = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = sorted([
        p for p in PHOTOS_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_exts and "__MACOSX" not in p.parts
    ])
    print(f"Device: {DEVICE} | Threads: {torch.get_num_threads()} | Found images: {len(image_paths)}", flush=True)
    if not image_paths:
        raise RuntimeError(f"No images found in {PHOTOS_DIR}")

    # ---------------- STAGE 1: Mask2Former Swin-Large ----------------
    stage1_predictions = defaultdict(list)
    t0 = time.time()
    print(f"[Stage 1] Loading {MASK2FORMER_MODEL_ID} ...", flush=True)
    m2f_processor = AutoImageProcessor.from_pretrained(MASK2FORMER_MODEL_ID)
    m2f_model = Mask2FormerForUniversalSegmentation.from_pretrained(MASK2FORMER_MODEL_ID).to(DEVICE, dtype=torch.float32).eval()
    m2f_id2label = m2f_model.config.id2label

    for idx, img_path in enumerate(image_paths, 1):
        pil_orig = load_image_exif_corrected(img_path)
        orig_w, orig_h = pil_orig.size
        min_area_px = max(25, int(orig_w * orig_h * MIN_MASK_AREA_RATIO))

        pil_inf, _ = resize_for_inference(pil_orig)
        inputs = m2f_processor(images=pil_inf, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(DEVICE, dtype=torch.float32)

        with torch.inference_mode():
            outputs = m2f_model(**inputs)

        panoptic_result = m2f_processor.post_process_panoptic_segmentation(
            outputs,
            threshold=M2F_THRESHOLD,
            mask_threshold=0.5,
            overlap_mask_area_threshold=0.5,
            target_sizes=[(orig_h, orig_w)],
        )[0]

        seg_map = panoptic_result["segmentation"].cpu().numpy()
        for seg_info in panoptic_result["segments_info"]:
            coco_label = m2f_id2label.get(int(seg_info["label_id"]), "").strip()
            target_label = COCO_PANOPTIC_TO_TARGET.get(coco_label)
            if target_label is None or target_label not in NAME_TO_CAT_ID:
                continue
            mask = (seg_map == int(seg_info["id"]))
            if int(mask.sum()) < min_area_px:
                continue
            stage1_predictions[img_path.name].append({
                "label": target_label,
                "score": float(seg_info["score"]),
                "mask": mask,
                "source": "mask2former",
            })

        if idx % 5 == 0 or idx == len(image_paths):
            print(f"[Stage 1] {idx}/{len(image_paths)} done ({time.time() - t0:.1f}s)", flush=True)

    stage1_time = time.time() - t0
    print(f"[Stage 1] Complete in {stage1_time:.1f}s | Segments: {sum(len(v) for v in stage1_predictions.values())}", flush=True)
    del m2f_model, m2f_processor
    gc.collect()

    # ---------------- STAGE 2: Grounding DINO Base + SAM 2.1 Large ----------------
    stage2_predictions = defaultdict(list)
    t1 = time.time()
    print(f"[Stage 2] Loading {GDINO_MODEL_ID} + {SAM2_WEIGHTS} ...", flush=True)
    gdino_processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL_ID).to(DEVICE, dtype=torch.float32).eval()
    sam2_model = SAM(SAM2_WEIGHTS)

    for idx, img_path in enumerate(image_paths, 1):
        pil_orig = load_image_exif_corrected(img_path)
        orig_w, orig_h = pil_orig.size
        min_area_px = max(25, int(orig_w * orig_h * MIN_MASK_AREA_RATIO))
        pil_inf, _ = resize_for_inference(pil_orig)

        detected_boxes, detected_labels, detected_scores = [], [], []
        for batch_map in GDINO_PROMPT_BATCHES:
            prompt_text = ". ".join(batch_map.keys()) + "."
            inputs = gdino_processor(images=pil_inf, text=prompt_text, return_tensors="pt").to(DEVICE)

            with torch.inference_mode():
                outputs = gdino_model(**inputs)

            post_kwargs = {
                "threshold": GDINO_BOX_THRESHOLD,
                "text_threshold": GDINO_TEXT_THRESHOLD,
                "target_sizes": [(orig_h, orig_w)],
            }
            try:
                res = gdino_processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, **post_kwargs
                )[0]
            except TypeError:
                post_kwargs["box_threshold"] = post_kwargs.pop("threshold")
                res = gdino_processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, **post_kwargs
                )[0]

            raw_labels = res.get("text_labels", res.get("labels", []))
            boxes = res["boxes"].detach().cpu().numpy()
            scores = res["scores"].detach().cpu().numpy()

            for box, score, raw_lbl in zip(boxes, scores, raw_labels):
                target_cls = match_phrase_to_target(str(raw_lbl), batch_map)
                if target_cls is None or target_cls not in NAME_TO_CAT_ID:
                    continue
                x1, y1, x2, y2 = box.tolist()
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(orig_w), x2), min(float(orig_h), y2)
                if (x2 - x1) < 4.0 or (y2 - y1) < 4.0:
                    continue
                detected_boxes.append([x1, y1, x2, y2])
                detected_labels.append(target_cls)
                detected_scores.append(float(score))

        if detected_boxes:
            img_bgr = cv2.cvtColor(np.array(pil_orig), cv2.COLOR_RGB2BGR)
            sam_results = sam2_model.predict(
                source=img_bgr,
                bboxes=detected_boxes,
                device=0 if DEVICE.type == "cuda" else "cpu",
                verbose=False,
            )
            if sam_results and sam_results[0].masks is not None:
                masks_np = sam_results[0].masks.data.detach().cpu().numpy()
                for mask_arr, lbl, sc in zip(masks_np, detected_labels, detected_scores):
                    if mask_arr.shape != (orig_h, orig_w):
                        mask_arr = cv2.resize(mask_arr.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    mask_bool = mask_arr > 0.5
                    if int(mask_bool.sum()) < min_area_px:
                        continue
                    stage2_predictions[img_path.name].append({
                        "label": lbl,
                        "score": sc,
                        "mask": mask_bool,
                        "source": "gdino_sam2",
                    })

        if idx % 5 == 0 or idx == len(image_paths):
            print(f"[Stage 2] {idx}/{len(image_paths)} done ({time.time() - t1:.1f}s)", flush=True)

    stage2_time = time.time() - t1
    print(f"[Stage 2] Complete in {stage2_time:.1f}s | Segments: {sum(len(v) for v in stage2_predictions.values())}", flush=True)
    del gdino_model, gdino_processor, sam2_model
    gc.collect()

    # ---------------- STAGE 3: Fuse, Export COCO JSON & Overlays ----------------
    coco_output = {
        "info": {
            "description": "SOTA Hybrid Auto-Annotations (Mask2Former Swin-L + Grounding DINO Base + SAM 2.1 Large)",
            "version": "1.0",
            "year": 2026,
        },
        "licenses": [],
        "categories": build_coco_categories(include_root=INCLUDE_ROBOFLOW_ROOT_CAT),
        "images": [],
        "annotations": [],
    }

    rng = np.random.default_rng(SEED)
    class_colors = {name: tuple(int(c) for c in rng.integers(40, 245, size=3)) for name in CLASS_NAMES_74}

    def render_overlay(pil_img: Image.Image, anns: list) -> np.ndarray:
        canvas = np.array(pil_img).copy()
        overlay = canvas.copy()
        for ann in anns:
            color = class_colors.get(ann["label"], (0, 255, 0))
            mask = ann["mask"]
            overlay[mask] = color
            contours, _ = cv2.findContours((mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, color, 2)

        blended = cv2.addWeighted(overlay, 0.38, canvas, 0.62, 0)
        for ann in anns:
            color = class_colors.get(ann["label"], (0, 255, 0))
            ys, xs = np.where(ann["mask"])
            if len(xs) == 0:
                continue
            cx, cy = int(np.median(xs)), int(np.median(ys))
            tag = f"{ann['label']} {ann['score']:.2f}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            x0 = max(2, min(blended.shape[1] - tw - 6, cx - tw // 2))
            y0 = max(th + 4, min(blended.shape[0] - 4, cy))
            cv2.rectangle(blended, (x0 - 2, y0 - th - 3), (x0 + tw + 2, y0 + 2), color, -1)
            cv2.putText(blended, tag, (x0, y0 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        return blended

    ann_id = 1
    per_image_rows = []
    class_counter = Counter()
    source_counter = Counter()

    for img_id, img_path in enumerate(image_paths, start=1):
        pil_orig = load_image_exif_corrected(img_path)
        orig_w, orig_h = pil_orig.size
        coco_output["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": orig_w,
            "height": orig_h,
        })

        combined = stage1_predictions[img_path.name] + stage2_predictions[img_path.name]
        fused = fuse_stage_predictions(combined, iou_thresh=MERGE_IOU_THRESHOLD)

        valid_fused = []
        for item in fused:
            polygons, bbox, area = mask_to_polygons_and_bbox(item["mask"])
            if not polygons or bbox is None:
                continue
            cat_id = NAME_TO_CAT_ID[item["label"]]
            coco_output["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [round(v, 2) for v in bbox],
                "area": round(area, 2),
                "segmentation": polygons,
                "iscrowd": 0,
                "score": round(float(item["score"]), 4),
                "source": item["source"],
            })
            ann_id += 1
            class_counter[item["label"]] += 1
            source_counter[item["source"]] += 1
            valid_fused.append(item)

        per_image_rows.append({
            "image_id": img_id,
            "file_name": img_path.name,
            "width": orig_w,
            "height": orig_h,
            "sota_total_instances": len(valid_fused),
            "sota_unique_classes": len({x["label"] for x in valid_fused}),
            "from_mask2former": sum(1 for x in valid_fused if x["source"] == "mask2former"),
            "from_gdino_sam2": sum(1 for x in valid_fused if x["source"] == "gdino_sam2"),
            "nrp9_book": sum(1 for x in valid_fused if x["label"] == "book"),
            "nrp9_light": sum(1 for x in valid_fused if x["label"] == "light"),
            "nrp9_spoon": sum(1 for x in valid_fused if x["label"] == "spoon"),
        })

        vis_rgb = render_overlay(pil_orig, valid_fused)
        cv2.imwrite(
            str(OUTPUT_DIR / "visualizations" / f"{img_path.stem}_sota.jpg"),
            cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )

    coco_json_path = OUTPUT_DIR / "sota_annotations.coco.json"
    with open(coco_json_path, "w", encoding="utf-8") as f:
        json.dump(coco_output, f, indent=2)

    pd.DataFrame(per_image_rows).to_csv(OUTPUT_DIR / "per_image_summary.csv", index=False)
    per_class_df = pd.DataFrame([
        {
            "category_id": NAME_TO_CAT_ID[cls],
            "class_name": cls,
            "taxonomy": "stuff" if cls in STUFF_CLASSES else "thing",
            "sota_instances": class_counter.get(cls, 0),
        }
        for cls in CLASS_NAMES_74
    ]).sort_values("sota_instances", ascending=False)
    per_class_df.to_csv(OUTPUT_DIR / "per_class_summary.csv", index=False)

    vis_files = sorted((OUTPUT_DIR / "visualizations").glob("*.jpg"))
    if vis_files:
        sample_files = vis_files[:6]
        cols = 2
        rows = math.ceil(len(sample_files) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(16, 6 * rows))
        axes = np.array(axes).reshape(-1)
        for ax, vf in zip(axes, sample_files):
            img = cv2.cvtColor(cv2.imread(str(vf)), cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.set_title(vf.name, fontsize=10)
            ax.axis("off")
        for ax in axes[len(sample_files):]:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "preview_grid.png", dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    bundle_path = BASE_DIR / "sota_auto_annotation_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(OUTPUT_DIR.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(OUTPUT_DIR))

    print(f"\nExported COCO JSON : {coco_json_path}", flush=True)
    print(f"Total images       : {len(coco_output['images'])}", flush=True)
    print(f"Total annotations  : {len(coco_output['annotations'])} (Mask2Former: {source_counter['mask2former']}, GDINO+SAM2: {source_counter['gdino_sam2']})", flush=True)
    print(f"Active classes     : {(per_class_df['sota_instances'] > 0).sum()} / 74", flush=True)
    print(f"Export bundle ready: {bundle_path} ({bundle_path.stat().st_size / 1024**2:.2f} MB)", flush=True)
    print("\nTop 15 detected classes:")
    print(per_class_df.head(15).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
