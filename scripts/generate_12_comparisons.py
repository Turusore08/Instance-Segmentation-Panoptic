from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ASSETS_DIR, MODEL_REGISTRY, ORIG_DATASET_DIR, load_orig_class_names  # noqa: E402
from src.inference import (  # noqa: E402
    GroundTruthObject,
    load_mask2former_model,
    load_maskrcnn_model,
    load_yolo_model,
    parse_yolo_seg_label,
    predict_mask2former,
    predict_maskrcnn,
    predict_yolo,
    render_gt_overlay,
    render_overlay,
)
from src.metrics import mask_iou  # noqa: E402

warnings.filterwarnings("ignore")

MANUAL_PHOTOS_DIR = (
    PROJECT_ROOT / "viskom-data-sendiri"
    if (PROJECT_ROOT / "viskom-data-sendiri").exists()
    else WORKSPACE_ROOT / "viskom-data-sendiri"
)
SOTA_COCO_PATH = WORKSPACE_ROOT / "sota_auto_annotation_bundle" / "sota_annotations.coco.json"
OUT_DIR = ASSETS_DIR / "segmentation_comparisons"
CONF_THRESH = 0.25


def mean_best_iou(preds: list, gts: List[GroundTruthObject]) -> float:
    if not gts:
        return 0.0
    return float(np.mean([max((mask_iou(g.mask, p.mask) for p in preds), default=0.0) for g in gts]))


def select_orig_samples(orig_names: Dict[int, str], count: int = 6) -> List[dict]:
    img_dir = ORIG_DATASET_DIR / "test" / "images"
    lbl_dir = ORIG_DATASET_DIR / "test" / "labels"
    candidates = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lf = lbl_dir / f"{p.stem}.txt"
        if not lf.exists():
            continue
        lines = [ln.strip().split() for ln in lf.read_text(encoding="utf-8").splitlines() if len(ln.strip().split()) >= 7]
        if not (5 <= len(lines) <= 16):
            continue
        uniq_cls = len({int(float(parts[0])) for parts in lines})
        candidates.append({
            "file_name": p.name,
            "path": p,
            "label_path": lf,
            "uniq_cls": uniq_cls,
            "n_anns": len(lines),
        })
    candidates.sort(key=lambda c: (-c["uniq_cls"], -c["n_anns"], c["file_name"]))
    # Pick evenly across top 24 diverse candidates so rooms/scenes vary cleanly
    pool = candidates[: max(count * 4, count)]
    step = max(1, len(pool) // count)
    picked = [pool[i * step] for i in range(count)]
    return picked


def select_manual_samples(sota_coco: dict, count: int = 6) -> List[dict]:
    anns_by_img = defaultdict(list)
    for a in sota_coco["annotations"]:
        anns_by_img[int(a["image_id"])].append(a)
    cat_map = {int(c["id"]): c["name"] for c in sota_coco["categories"]}

    candidates = []
    for im in sota_coco["images"]:
        iid = int(im["id"])
        anns = anns_by_img[iid]
        if not anns:
            continue
        p = MANUAL_PHOTOS_DIR / im["file_name"]
        if not p.exists():
            matches = list(MANUAL_PHOTOS_DIR.rglob(im["file_name"]))
            if not matches:
                raise FileNotFoundError(f"Manual photo not found: {im['file_name']}")
            p = matches[0]
        uniq_cls = len({int(a["category_id"]) for a in anns})
        candidates.append({
            "image_id": iid,
            "file_name": im["file_name"],
            "width": int(im["width"]),
            "height": int(im["height"]),
            "path": p,
            "anns": anns,
            "cat_map": cat_map,
            "uniq_cls": uniq_cls,
            "n_anns": len(anns),
        })
    candidates.sort(key=lambda c: (-c["uniq_cls"], -c["n_anns"], c["file_name"]))
    return candidates[:count]


def rasterize_sota_gts(anns: List[dict], cat_map: Dict[int, str], w: int, h: int) -> List[GroundTruthObject]:
    gts: List[GroundTruthObject] = []
    for a in anns:
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in a["segmentation"]:
            pts = np.round(np.array(poly, dtype=np.float32).reshape(-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
        area_px = int(mask.sum())
        if area_px <= 4:
            continue
        ys, xs = np.where(mask > 0)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        sota_cid = int(a["category_id"])
        gts.append(
            GroundTruthObject(
                category_id=sota_cid - 1,
                class_name=cat_map.get(sota_cid, f"cls_{sota_cid}"),
                mask=mask,
                bbox=bbox,
                area_px=area_px,
            )
        )
    return gts


def save_4panel_figure(
    out_path: Path,
    title_header: str,
    img_rgb: np.ndarray,
    gts: List[GroundTruthObject],
    m2f_preds: list,
    yolo_preds: list,
    rcnn_preds: list,
    is_portrait: bool,
) -> dict:
    gt_vis = render_gt_overlay(img_rgb, gts, alpha=0.40)
    m2f_vis = render_overlay(img_rgb, m2f_preds, alpha=0.45)
    yolo_vis = render_overlay(img_rgb, yolo_preds, alpha=0.45)
    rcnn_vis = render_overlay(img_rgb, rcnn_preds, alpha=0.45)

    m2f_iou = mean_best_iou(m2f_preds, gts)
    yolo_iou = mean_best_iou(yolo_preds, gts)
    rcnn_iou = mean_best_iou(rcnn_preds, gts)

    figsize = (16, 7.2) if is_portrait else (18, 5.2)
    fig, axes = plt.subplots(1, 4, figsize=figsize, facecolor="white")
    panels = [
        (gt_vis, f"Ground Truth ({len(gts)} objek)"),
        (m2f_vis, f"Mask2Former-Large Skn B\n({len(m2f_preds)} det | mIoU={m2f_iou:.3f})"),
        (yolo_vis, f"YOLOv26l-seg Skn B\n({len(yolo_preds)} det | mIoU={yolo_iou:.3f})"),
        (rcnn_vis, f"Mask-RCNN-Large Skn C\n({len(rcnn_preds)} det | mIoU={rcnn_iou:.3f})"),
    ]
    for ax, (vis, sub) in zip(axes, panels):
        ax.set_facecolor("white")
        ax.imshow(vis)
        ax.set_title(sub, fontsize=10, fontweight="bold", color="black", pad=6)
        ax.axis("off")

    fig.suptitle(title_header, fontsize=11.5, fontweight="bold", color="black", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)

    return {
        "out_file": out_path.name,
        "n_gt": len(gts),
        "m2f_det": len(m2f_preds),
        "m2f_miou": round(m2f_iou, 4),
        "yolo_det": len(yolo_preds),
        "yolo_miou": round(yolo_iou, 4),
        "rcnn_det": len(rcnn_preds),
        "rcnn_miou": round(rcnn_iou, 4),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    orig_names = load_orig_class_names()
    sota_coco = json.loads(SOTA_COCO_PATH.read_text(encoding="utf-8"))

    orig_picks = select_orig_samples(orig_names, count=6)
    manual_picks = select_manual_samples(sota_coco, count=6)

    print("Loading 3 best Large checkpoints...")
    m2f_pm = load_mask2former_model(MODEL_REGISTRY["Mask2Former-Large"]["checkpoints"]["B"])
    yolo_m = load_yolo_model(MODEL_REGISTRY["YOLOv26l-seg"]["checkpoints"]["B"])
    rcnn_m = load_maskrcnn_model(MODEL_REGISTRY["Mask-RCNN-Large"]["checkpoints"]["C"], num_classes=len(orig_names))

    summary = []

    for idx, item in enumerate(orig_picks, 1):
        with Image.open(item["path"]) as im:
            img_rgb = np.array(im.convert("RGB"))
        h, w = img_rgb.shape[:2]
        gts = parse_yolo_seg_label(item["label_path"].read_text(encoding="utf-8"), w, h, orig_names)
        m2f_p, _ = predict_mask2former(m2f_pm, img_rgb, conf_thresh=CONF_THRESH, class_names=orig_names)
        yolo_p, _ = predict_yolo(yolo_m, img_rgb, conf_thresh=CONF_THRESH, class_names=orig_names)
        rcnn_p, _ = predict_maskrcnn(rcnn_m, img_rgb, conf_thresh=CONF_THRESH, class_names=orig_names)

        out_path = OUT_DIR / f"orig_{idx:02d}.png"
        info = save_4panel_figure(
            out_path,
            f"TEST SET ASLI #{idx}: {item['file_name']} ({w}x{h}, conf >= {CONF_THRESH})",
            img_rgb, gts, m2f_p, yolo_p, rcnn_p, is_portrait=(h > w),
        )
        info["source"] = "orig_test"
        info["source_file"] = item["file_name"]
        summary.append(info)
        print(f"Saved {out_path.name}: {info}", flush=True)

    for idx, item in enumerate(manual_picks, 1):
        with Image.open(item["path"]) as im:
            img_rgb = np.array(ImageOps.exif_transpose(im).convert("RGB"))
        h, w = img_rgb.shape[:2]
        assert (w, h) == (item["width"], item["height"]), f"Dimension mismatch on {item['file_name']}"
        gts = rasterize_sota_gts(item["anns"], item["cat_map"], w, h)
        m2f_p, _ = predict_mask2former(m2f_pm, img_rgb, conf_thresh=CONF_THRESH, class_names=orig_names)
        yolo_p, _ = predict_yolo(yolo_m, img_rgb, conf_thresh=CONF_THRESH, class_names=orig_names)
        rcnn_p, _ = predict_maskrcnn(rcnn_m, img_rgb, conf_thresh=CONF_THRESH, class_names=orig_names)

        out_path = OUT_DIR / f"manual_{idx:02d}.png"
        info = save_4panel_figure(
            out_path,
            f"DATASET MANUAL #{idx}: {item['file_name']} ({w}x{h}, conf >= {CONF_THRESH})",
            img_rgb, gts, m2f_p, yolo_p, rcnn_p, is_portrait=(h > w),
        )
        info["source"] = "manual_sota"
        info["source_file"] = item["file_name"]
        summary.append(info)
        print(f"Saved {out_path.name}: {info}", flush=True)

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDONE! Generated {len(summary)} comparison PNGs in {OUT_DIR}")


if __name__ == "__main__":
    main()
