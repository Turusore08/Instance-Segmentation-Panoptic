from __future__ import annotations

import json
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageOps
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ASSETS_DIR, MODEL_REGISTRY, ORIG_DATASET_DIR, load_orig_class_names  # noqa: E402
from src.inference import (  # noqa: E402
    load_mask2former_model,
    load_maskrcnn_model,
    load_yolo_model,
    predict_mask2former,
    predict_maskrcnn,
    predict_yolo,
    preprocess_domain_alignment,
)

warnings.filterwarnings("ignore")

MANUAL_PHOTOS_DIR = (
    PROJECT_ROOT / "viskom-data-sendiri"
    if (PROJECT_ROOT / "viskom-data-sendiri").exists()
    else WORKSPACE_ROOT / "viskom-data-sendiri"
)
SOTA_COCO_PATH = WORKSPACE_ROOT / "sota_auto_annotation_bundle" / "sota_annotations.coco.json"
OUT_JSON_PATH = ASSETS_DIR / "datashift_metrics.json"


def encode_rle(bin_mask: np.ndarray) -> dict:
    rle = mask_util.encode(np.asfortranarray(bin_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def run_cocoeval(gt_dict: dict, preds: List[dict], per_class_names: Dict[int, str] | None = None) -> Dict[str, Any]:
    tmp_gt = ASSETS_DIR / "_tmp_datashift_gt.json"
    with open(tmp_gt, "w", encoding="utf-8") as f:
        json.dump(gt_dict, f)
    coco_gt = COCO(str(tmp_gt))

    if not preds:
        res: Dict[str, Any] = {
            "mAP50_95": 0.0,
            "mAP50": 0.0,
            "mAP75": 0.0,
            "AP_small": 0.0,
            "AP_medium": 0.0,
            "AP_large": 0.0,
            "mAR100": 0.0,
            "num_preds": 0,
            "per_class": {},
        }
        tmp_gt.unlink(missing_ok=True)
        return res

    coco_dt = coco_gt.loadRes(preds)
    ev = COCOeval(coco_gt, coco_dt, iouType="segm")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    def _clean(val: float) -> float | None:
        return None if val < 0 else round(float(val), 4)

    out: Dict[str, Any] = {
        "mAP50_95": round(float(max(0.0, ev.stats[0])), 4),
        "mAP50": round(float(max(0.0, ev.stats[1])), 4),
        "mAP75": round(float(max(0.0, ev.stats[2])), 4),
        "AP_small": _clean(ev.stats[3]),
        "AP_medium": _clean(ev.stats[4]),
        "AP_large": _clean(ev.stats[5]),
        "mAR100": round(float(max(0.0, ev.stats[8])), 4),
        "num_preds": len(preds),
        "per_class": {},
    }

    if per_class_names is not None:
        prec = ev.eval["precision"]  # [T, R, K, A, M]
        rec = ev.eval["recall"]      # [T, K, A, M]
        for k_idx, cid in enumerate(ev.params.catIds):
            p50 = prec[0, :, k_idx, 0, 2]
            pall = prec[:, :, k_idx, 0, 2]
            rall = rec[:, k_idx, 0, 2]
            v50 = p50[p50 > -1]
            vall = pall[pall > -1]
            vr = rall[rall > -1]
            if len(v50) == 0 and len(vall) == 0:
                continue
            cname = per_class_names.get(int(cid), f"cls_{cid}")
            out["per_class"][cname] = {
                "category_id_sota": int(cid),
                "category_id_model": int(cid) - 1,
                "ap50": round(float(np.mean(v50)), 4) if len(v50) > 0 else None,
                "ap50_95": round(float(np.mean(vall)), 4) if len(vall) > 0 else None,
                "ar100": round(float(np.mean(vr)), 4) if len(vr) > 0 else None,
            }

    tmp_gt.unlink(missing_ok=True)
    return out


def compute_dataset_profile(orig_per_class_path: Path, sota_coco: dict) -> Dict[str, Any]:
    # Original Panoptic Indoor dataset stats
    orig_splits: Dict[str, Dict[str, int]] = {}
    for sp in ["train", "valid", "test"]:
        img_dir = ORIG_DATASET_DIR / sp / "images"
        lbl_dir = ORIG_DATASET_DIR / sp / "labels"
        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        anns = sum(
            sum(1 for ln in (lbl_dir / f"{p.stem}.txt").read_text(encoding="utf-8").splitlines() if len(ln.strip().split()) >= 7)
            for p in imgs
            if (lbl_dir / f"{p.stem}.txt").exists()
        )
        orig_splits[sp] = {"images": len(imgs), "annotations": anns}

    # Manual SOTA dataset stats
    resolutions = Counter()
    areas = [float(a["area"]) for a in sota_coco["annotations"]]
    gt_counts_by_cat = Counter(int(a["category_id"]) for a in sota_coco["annotations"])
    cat_id_to_name = {int(c["id"]): c["name"] for c in sota_coco["categories"]}
    rgb_count = 0
    for im_info in sota_coco["images"]:
        resolutions[f"{im_info['width']}x{im_info['height']}"] += 1
        p = MANUAL_PHOTOS_DIR / im_info["file_name"]
        with Image.open(p) as im:
            arr = np.array(ImageOps.exif_transpose(im).convert("RGB"))
            if not (np.array_equal(arr[:, :, 0], arr[:, :, 1]) and np.array_equal(arr[:, :, 1], arr[:, :, 2])):
                rgb_count += 1

    small_cnt = sum(1 for a in areas if a < 32 ** 2)
    med_cnt = sum(1 for a in areas if 32 ** 2 <= a < 96 ** 2)
    large_cnt = sum(1 for a in areas if a >= 96 ** 2)

    return {
        "orig_panoptic_indoor": orig_splits,
        "manual_sota": {
            "images": len(sota_coco["images"]),
            "annotations": len(sota_coco["annotations"]),
            "active_classes": len(gt_counts_by_cat),
            "true_rgb_images": rgb_count,
            "resolutions": dict(resolutions),
            "area_scale_counts": {"small": small_cnt, "medium": med_cnt, "large": large_cnt},
            "gt_counts_by_class": {cat_id_to_name[k]: int(v) for k, v in sorted(gt_counts_by_cat.items(), key=lambda kv: -kv[1])},
        },
    }


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    orig_names = load_orig_class_names()
    sota_coco = json.loads(SOTA_COCO_PATH.read_text(encoding="utf-8"))

    # Verify all image files exist
    image_lookup: Dict[int, Path] = {}
    for im in sota_coco["images"]:
        p = MANUAL_PHOTOS_DIR / im["file_name"]
        if not p.exists():
            matches = list(MANUAL_PHOTOS_DIR.rglob(im["file_name"]))
            if not matches:
                raise FileNotFoundError(f"Missing manual photo: {im['file_name']} under {MANUAL_PHOTOS_DIR}")
            p = matches[0]
        image_lookup[int(im["id"])] = p

    sota_id_to_name = {int(c["id"]): c["name"] for c in sota_coco["categories"]}

    gt_74 = {
        "info": sota_coco.get("info", {}),
        "licenses": [],
        "categories": sota_coco["categories"],
        "images": sota_coco["images"],
        "annotations": [
            {
                "id": int(a["id"]),
                "image_id": int(a["image_id"]),
                "category_id": int(a["category_id"]),
                "bbox": a["bbox"],
                "area": a["area"],
                "segmentation": a["segmentation"],
                "iscrowd": int(a.get("iscrowd", 0)),
            }
            for a in sota_coco["annotations"]
        ],
    }

    gt_agn = {
        "info": sota_coco.get("info", {}),
        "licenses": [],
        "categories": [{"id": 1, "name": "object"}],
        "images": sota_coco["images"],
        "annotations": [{**a, "category_id": 1} for a in gt_74["annotations"]],
    }

    print("Loading best Large checkpoints...")
    yolo_m = load_yolo_model(MODEL_REGISTRY["YOLOv26l-seg"]["checkpoints"]["B"])
    m2f_pm = load_mask2former_model(MODEL_REGISTRY["Mask2Former-Large"]["checkpoints"]["B"])
    rcnn_m = load_maskrcnn_model(MODEL_REGISTRY["Mask-RCNN-Large"]["checkpoints"]["C"], num_classes=len(orig_names))

    model_keys = ["Mask2Former-Large", "YOLOv26l-seg", "Mask-RCNN-Large"]
    preds_rgb_74: Dict[str, List[dict]] = {m: [] for m in model_keys}
    preds_rgb_agn: Dict[str, List[dict]] = {m: [] for m in model_keys}
    preds_clahe_74: Dict[str, List[dict]] = {m: [] for m in model_keys}
    preds_25_counts_rgb: Dict[str, int] = {m: 0 for m in model_keys}
    preds_25_counts_clahe: Dict[str, int] = {m: 0 for m in model_keys}

    for idx, im_info in enumerate(sota_coco["images"], 1):
        iid = int(im_info["id"])
        p = image_lookup[iid]
        with Image.open(p) as im:
            img_rgb = np.array(ImageOps.exif_transpose(im).convert("RGB"))
        assert (img_rgb.shape[1], img_rgb.shape[0]) == (im_info["width"], im_info["height"]), (
            f"Shape mismatch for {p.name}: {(img_rgb.shape[1], img_rgb.shape[0])} vs {(im_info['width'], im_info['height'])}"
        )
        img_clahe = preprocess_domain_alignment(img_rgb, True)

        for cond_name, img_in, dest_74, dest_agn, cnt25 in [
            ("rgb", img_rgb, preds_rgb_74, preds_rgb_agn, preds_25_counts_rgb),
            ("clahe", img_clahe, preds_clahe_74, None, preds_25_counts_clahe),
        ]:
            m_dets, _ = predict_mask2former(m2f_pm, img_in, conf_thresh=0.05, class_names=orig_names)
            y_dets, _ = predict_yolo(yolo_m, img_in, conf_thresh=0.05, class_names=orig_names)
            r_dets, _ = predict_maskrcnn(rcnn_m, img_in, conf_thresh=0.05, class_names=orig_names)

            for mname, dets in [("Mask2Former-Large", m_dets), ("YOLOv26l-seg", y_dets), ("Mask-RCNN-Large", r_dets)]:
                for d in dets:
                    rle = encode_rle(d.mask)
                    # Model uses 0..73 while SOTA COCO categories use 1..74
                    dest_74[mname].append({
                        "image_id": iid,
                        "category_id": int(d.category_id) + 1,
                        "segmentation": rle,
                        "score": float(d.score),
                    })
                    if dest_agn is not None:
                        dest_agn[mname].append({
                            "image_id": iid,
                            "category_id": 1,
                            "segmentation": rle,
                            "score": float(d.score),
                        })
                    if d.score >= 0.25:
                        cnt25[mname] += 1

        if idx % 10 == 0 or idx == len(sota_coco["images"]):
            print(f"Evaluated {idx}/{len(sota_coco['images'])} manual images (RGB + Grayscale CLAHE)...", flush=True)

    results = {
        "dataset_profile": compute_dataset_profile(ASSETS_DIR / "per_class_ap.json", sota_coco),
        "manual_rgb_74_exact": {},
        "manual_rgb_class_agnostic": {},
        "manual_clahe_74_exact": {},
    }

    for mname in model_keys:
        print(f"\n=== COCOeval Manual RGB 74-Class: {mname} ===")
        res_74 = run_cocoeval(gt_74, preds_rgb_74[mname], per_class_names=sota_id_to_name)
        res_74["num_preds_25"] = preds_25_counts_rgb[mname]
        results["manual_rgb_74_exact"][mname] = res_74

        print(f"\n=== COCOeval Manual RGB Class-Agnostic: {mname} ===")
        res_agn = run_cocoeval(gt_agn, preds_rgb_agn[mname])
        results["manual_rgb_class_agnostic"][mname] = res_agn

        print(f"\n=== COCOeval Manual Grayscale+CLAHE 74-Class: {mname} ===")
        res_cl = run_cocoeval(gt_74, preds_clahe_74[mname], per_class_names=sota_id_to_name)
        res_cl["num_preds_25"] = preds_25_counts_clahe[mname]
        results["manual_clahe_74_exact"][mname] = res_cl

    OUT_JSON_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved datashift metrics to {OUT_JSON_PATH}")


if __name__ == "__main__":
    main()
