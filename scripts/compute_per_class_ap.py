from __future__ import annotations

import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    ASSETS_DIR,
    MODEL_REGISTRY,
    ORIG_DATASET_DIR,
    PER_CLASS_AP_PATH,
    load_orig_class_names,
)
from src.inference import (  # noqa: E402
    load_mask2former_model,
    load_maskrcnn_model,
    load_yolo_model,
    predict_mask2former,
    predict_maskrcnn,
    predict_yolo,
)

warnings.filterwarnings("ignore")


def encode_rle(bin_mask: np.ndarray) -> dict:
    rle = mask_util.encode(np.asfortranarray(bin_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def build_orig_test_coco(max_images: int = 60) -> tuple[dict, dict[int, int]]:
    class_names = load_orig_class_names()
    img_dir = ORIG_DATASET_DIR / "test" / "images"
    lbl_dir = ORIG_DATASET_DIR / "test" / "labels"
    imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])[:max_images]
    categories = [{"id": k, "name": v} for k, v in class_names.items()]
    images, anns = [], []
    gt_counts = Counter()
    ann_id = 1
    for img_id, p in enumerate(imgs, 1):
        with Image.open(p) as im:
            w, h = im.size
        images.append({"id": img_id, "file_name": p.name, "path": str(p), "width": w, "height": h})
        lf = lbl_dir / f"{p.stem}.txt"
        if lf.exists():
            for line in lf.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
                    continue
                cid = int(float(parts[0]))
                coords = np.clip(np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2), 0.0, 1.0)
                coords[:, 0] *= w
                coords[:, 1] *= h
                x1, y1 = float(coords[:, 0].min()), float(coords[:, 1].min())
                x2, y2 = float(coords[:, 0].max()), float(coords[:, 1].max())
                if x2 - x1 <= 1 or y2 - y1 <= 1:
                    continue
                area = max(float(cv2.contourArea(coords)), (x2 - x1) * (y2 - y1))
                anns.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cid,
                    "segmentation": [[round(float(v), 2) for v in coords.flatten().tolist()]],
                    "area": round(area, 2),
                    "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                    "iscrowd": 0,
                })
                gt_counts[cid] += 1
                ann_id += 1
    return {"info": {}, "licenses": [], "categories": categories, "images": images, "annotations": anns}, dict(gt_counts)


def safe_mean(arr: np.ndarray) -> float | None:
    valid = arr[arr > -1]
    if len(valid) == 0:
        return None
    return round(float(np.mean(valid)), 4)


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    class_names = load_orig_class_names()
    gt_dict, gt_counts = build_orig_test_coco(max_images=60)
    tmp_gt_path = ASSETS_DIR / "_tmp_orig60_gt.json"
    with open(tmp_gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_dict, f)
    coco_gt = COCO(str(tmp_gt_path))

    best_models = [
        ("YOLOv26l-seg", "B", "yolo", MODEL_REGISTRY["YOLOv26l-seg"]["checkpoints"]["B"]),
        ("Mask2Former-Large", "B", "mask2former", MODEL_REGISTRY["Mask2Former-Large"]["checkpoints"]["B"]),
        ("Mask-RCNN-Large", "C", "maskrcnn", MODEL_REGISTRY["Mask-RCNN-Large"]["checkpoints"]["C"]),
    ]

    preds_by_model: dict[str, list[dict]] = {m[0]: [] for m in best_models}
    pred_counts_05: dict[str, Counter] = {m[0]: Counter() for m in best_models}
    pred_counts_25: dict[str, Counter] = {m[0]: Counter() for m in best_models}

    print("Loading best Large models for per-class AP computation...")
    yolo_m = load_yolo_model(best_models[0][3])
    m2f_pm = load_mask2former_model(best_models[1][3])
    rcnn_m = load_maskrcnn_model(best_models[2][3], num_classes=len(class_names))

    for idx, im_info in enumerate(gt_dict["images"], 1):
        img_bgr = cv2.imread(im_info["path"])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        iid = im_info["id"]

        y_dets, _ = predict_yolo(yolo_m, img_rgb, conf_thresh=0.05, class_names=class_names)
        m_dets, _ = predict_mask2former(m2f_pm, img_rgb, conf_thresh=0.05, class_names=class_names)
        r_dets, _ = predict_maskrcnn(rcnn_m, img_rgb, conf_thresh=0.05, class_names=class_names)

        for mname, dets in [("YOLOv26l-seg", y_dets), ("Mask2Former-Large", m_dets), ("Mask-RCNN-Large", r_dets)]:
            for d in dets:
                preds_by_model[mname].append({
                    "image_id": iid,
                    "category_id": d.category_id,
                    "segmentation": encode_rle(d.mask),
                    "score": d.score,
                })
                pred_counts_05[mname][d.category_id] += 1
                if d.score >= 0.25:
                    pred_counts_25[mname][d.category_id] += 1
        if idx % 15 == 0:
            print(f"  Evaluated {idx}/{len(gt_dict['images'])} images...")

    recall_grid = [round(float(r), 2) for r in np.linspace(0.0, 1.0, 101)]
    out_data: dict = {
        "metadata": {
            "dataset": "panoptic-indoor-segmentation.v1i.yolov8-obb/test",
            "num_images": len(gt_dict["images"]),
            "num_total_annotations": len(gt_dict["annotations"]),
            "num_classes_total": len(class_names),
            "num_active_gt_classes": len(gt_counts),
            "recall_grid": recall_grid,
        },
        "models": {},
        "classes": {},
    }

    for cid, cname in class_names.items():
        out_data["classes"][str(cid)] = {
            "id": cid,
            "name": cname,
            "gt_count": int(gt_counts.get(cid, 0)),
            "metrics": {},
        }

    for mname, scen, _, _ in best_models:
        coco_dt = coco_gt.loadRes(preds_by_model[mname])
        ev = COCOeval(coco_gt, coco_dt, iouType="segm")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

        out_data["models"][mname] = {
            "scenario": scen,
            "mAP50_95": round(float(ev.stats[0]), 4),
            "mAP50": round(float(ev.stats[1]), 4),
            "mAP75": round(float(ev.stats[2]), 4),
            "mAR100": round(float(ev.stats[8]), 4),
            "total_preds_05": len(preds_by_model[mname]),
            "total_preds_25": int(sum(pred_counts_25[mname].values())),
        }

        prec_tensor = ev.eval["precision"]  # [T=10, R=101, K=74, A=4, M=3]
        rec_tensor = ev.eval["recall"]      # [T=10, K=74, A=4, M=3]
        cat_ids = ev.params.catIds

        for k_idx, cid in enumerate(cat_ids):
            ap50 = safe_mean(prec_tensor[0, :, k_idx, 0, 2])
            ap75 = safe_mean(prec_tensor[5, :, k_idx, 0, 2])
            ap50_95 = safe_mean(prec_tensor[:, :, k_idx, 0, 2])
            ar100 = safe_mean(rec_tensor[:, k_idx, 0, 2])
            pr_curve = [round(float(x), 4) if x > -1 else None for x in prec_tensor[0, :, k_idx, 0, 2]]
            out_data["classes"][str(cid)]["metrics"][mname] = {
                "ap50": ap50,
                "ap75": ap75,
                "ap50_95": ap50_95,
                "ar100": ar100,
                "pred_count_05": int(pred_counts_05[mname].get(cid, 0)),
                "pred_count_25": int(pred_counts_25[mname].get(cid, 0)),
                "pr_curve_50": pr_curve,
            }

    if tmp_gt_path.exists():
        tmp_gt_path.unlink()

    with open(PER_CLASS_AP_PATH, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved per-class AP and PR curves to {PER_CLASS_AP_PATH}")


if __name__ == "__main__":
    main()
