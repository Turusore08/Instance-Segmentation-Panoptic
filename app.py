from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageOps

from src.config import (
    CANONICAL_GROUPS,
    IMG_SIZE,
    MANUAL_TO_CANON,
    MODEL_REGISTRY,
    ORIG_TO_CANON,
    SCENARIO_LABELS,
    list_sample_images,
    load_manual_class_names,
    load_orig_class_names,
)
from src.inference import (
    Detection,
    GroundTruthObject,
    get_device,
    load_mask2former_model,
    load_maskrcnn_model,
    load_yolo_model,
    parse_yolo_seg_label,
    predict_mask2former,
    predict_maskrcnn,
    predict_yolo,
    preprocess_domain_alignment,
    render_gt_overlay,
    render_overlay,
    render_single_class_isolation,
)
from src.metrics import (
    compute_single_image_class_metrics,
    load_gpu_benchmark,
    load_per_class_ap_data,
    load_tflite_export_report,
    load_training_log,
)

st.set_page_config(
    page_title="Indoor Instance Segmentation Explorer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

MIN_CACHE_CONF = 0.05


@st.cache_data
def get_orig_names() -> Dict[int, str]:
    return load_orig_class_names()


@st.cache_data
def get_manual_names() -> Dict[int, str]:
    return load_manual_class_names()


@st.cache_data
def get_per_class_ap_cached() -> Optional[Dict[str, Any]]:
    return load_per_class_ap_data()


@st.cache_data
def get_benchmark_df() -> pd.DataFrame:
    return load_gpu_benchmark()


@st.cache_data
def get_tflite_report() -> List[Dict[str, Any]]:
    return load_tflite_export_report()


@st.cache_resource(max_entries=3)
def get_loaded_model(model_name: str, scenario: str) -> Any:
    spec = MODEL_REGISTRY[model_name]
    family = spec["family"]
    ckpt = spec["checkpoints"][scenario]
    if family == "yolo":
        return load_yolo_model(ckpt)
    if family == "mask2former":
        return load_mask2former_model(ckpt)
    if family == "maskrcnn":
        return load_maskrcnn_model(ckpt, num_classes=len(get_orig_names()))
    raise ValueError(f"Unknown model family: {family}")


def run_cached_inference(
    img_rgb: np.ndarray,
    img_hash: str,
    model_name: str,
    scenario: str,
    yolo_imgsz: int,
    yolo_iou: float,
    use_clahe: bool,
) -> Tuple[List[Detection], float, np.ndarray]:
    if "inference_cache" not in st.session_state:
        st.session_state["inference_cache"] = {}
    cache: Dict[tuple, Tuple[List[Detection], float, np.ndarray]] = st.session_state["inference_cache"]

    cache_key = (img_hash, model_name, scenario, yolo_imgsz if model_name == "YOLOv26l-seg" else IMG_SIZE, round(yolo_iou, 2), use_clahe)
    if cache_key in cache:
        return cache[cache_key]

    proc_img = preprocess_domain_alignment(img_rgb, use_clahe)
    loaded = get_loaded_model(model_name, scenario)
    family = MODEL_REGISTRY[model_name]["family"]
    class_names = get_orig_names()

    if family == "yolo":
        dets, elapsed_ms = predict_yolo(
            loaded, proc_img, conf_thresh=MIN_CACHE_CONF, iou_thresh=yolo_iou, imgsz=yolo_imgsz, class_names=class_names
        )
    elif family == "mask2former":
        dets, elapsed_ms = predict_mask2former(loaded, proc_img, conf_thresh=MIN_CACHE_CONF, class_names=class_names)
    else:
        dets, elapsed_ms = predict_maskrcnn(loaded, proc_img, conf_thresh=MIN_CACHE_CONF, class_names=class_names)

    if len(cache) >= 12:
        oldest_key = next(iter(cache))
        del cache[oldest_key]
    cache[cache_key] = (dets, elapsed_ms, proc_img)
    return dets, elapsed_ms, proc_img


def load_image_and_gt_ui(key_prefix: str) -> Tuple[Optional[np.ndarray], str, str, List[GroundTruthObject], bool]:
    """Render shared image selector (Upload vs Original Test vs Manual Test) and return (img_rgb, name, sha1, gts, is_manual_gt)."""
    source_mode = st.radio(
        "Sumber Foto",
        ["Upload Foto Baru", "Sampel Test Set Asli (432x432 Grayscale)", "Sampel Foto Manual (RGB Smartphone)"],
        horizontal=True,
        key=f"{key_prefix}_src_mode",
    )

    img_rgb: Optional[np.ndarray] = None
    img_name = ""
    img_hash = ""
    gts: List[GroundTruthObject] = []
    is_manual_gt = False

    if source_mode == "Upload Foto Baru":
        col_up1, col_up2 = st.columns([2, 1])
        with col_up1:
            uploaded = st.file_uploader(
                "Unggah foto ruangan indoor (.jpg, .jpeg, .png, .webp)",
                type=["jpg", "jpeg", "png", "webp"],
                key=f"{key_prefix}_uploader",
            )
        with col_up2:
            uploaded_lbl = st.file_uploader(
                "Opsional: Upload label Ground Truth YOLO (.txt) untuk hitung AP/IoU satu gambar",
                type=["txt"],
                key=f"{key_prefix}_lbl_uploader",
            )
        if uploaded is not None:
            raw_bytes = uploaded.getvalue()
            img_hash = hashlib.sha1(raw_bytes).hexdigest()[:16]
            img_name = uploaded.name
            pil_img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw_bytes))).convert("RGB")
            img_rgb = np.array(pil_img)
            if uploaded_lbl is not None:
                lbl_text = uploaded_lbl.getvalue().decode("utf-8", errors="ignore")
                gts = parse_yolo_seg_label(lbl_text, img_rgb.shape[1], img_rgb.shape[0], get_orig_names())
    elif source_mode == "Sampel Test Set Asli (432x432 Grayscale)":
        samples = list_sample_images("orig", limit=60)
        if not samples:
            st.warning("Folder sampel test set asli tidak ditemukan di workspace lokal.")
        else:
            labels = [s["label"] for s in samples]
            chosen_lbl = st.selectbox("Pilih Foto Test Set Asli", labels, index=0, key=f"{key_prefix}_orig_sel")
            chosen = samples[labels.index(chosen_lbl)]
            img_path = Path(chosen["image_path"])
            raw_bytes = img_path.read_bytes()
            img_hash = hashlib.sha1(raw_bytes).hexdigest()[:16]
            img_name = img_path.name
            img_rgb = np.array(Image.open(img_path).convert("RGB"))
            if chosen["label_path"]:
                lbl_text = Path(chosen["label_path"]).read_text(encoding="utf-8")
                gts = parse_yolo_seg_label(lbl_text, img_rgb.shape[1], img_rgb.shape[0], get_orig_names())
    else:
        samples = list_sample_images("manual", limit=40)
        if not samples:
            st.warning("Folder foto manual tidak ditemukan di workspace lokal.")
        else:
            labels = [s["label"] for s in samples]
            chosen_lbl = st.selectbox("Pilih Foto Manual", labels, index=0, key=f"{key_prefix}_man_sel")
            chosen = samples[labels.index(chosen_lbl)]
            img_path = Path(chosen["image_path"])
            raw_bytes = img_path.read_bytes()
            img_hash = hashlib.sha1(raw_bytes).hexdigest()[:16]
            img_name = img_path.name
            img_rgb = np.array(ImageOps.exif_transpose(Image.open(img_path)).convert("RGB"))
            if chosen["label_path"]:
                lbl_text = Path(chosen["label_path"]).read_text(encoding="utf-8")
                gts = parse_yolo_seg_label(lbl_text, img_rgb.shape[1], img_rgb.shape[0], get_manual_names())
                is_manual_gt = True

    return img_rgb, img_name, img_hash, gts, is_manual_gt


def encode_png_bytes(img_rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG")
    return buf.getvalue()


# ======================== HEADER & SIDEBAR ========================
st.title("🏠 Indoor Instance Segmentation Explorer (Large Models)")
st.caption(
    f"Workspace: Instance-Segmentation-Panoptic | Device Aktif: **{get_device().upper()}** | "
    "74 Kelas Indoor COCO-Panoptic | YOLOv26l-seg • Mask2Former-Large • Mask-RCNN-Large"
)

with st.sidebar:
    st.header("⚙️ Pengaturan Model & Inferensi")
    selected_model = st.selectbox(
        "Pilih Arsitektur Model",
        list(MODEL_REGISTRY.keys()),
        index=1,  # Default Mask2Former-Large
    )
    default_scen = str(MODEL_REGISTRY[selected_model]["default_scenario"])
    scen_keys = ["A", "B", "C"]
    selected_scenario = st.selectbox(
        "Pilih Skenario Augmentasi",
        scen_keys,
        index=scen_keys.index(default_scen),
        format_func=lambda s: SCENARIO_LABELS[s],
    )
    st.caption(str(MODEL_REGISTRY[selected_model]["description"]))

    ckpt_path = MODEL_REGISTRY[selected_model]["checkpoints"][selected_scenario]
    if not Path(ckpt_path).exists():
        st.error(f"Checkpoint tidak ditemukan: {ckpt_path}")

    st.divider()
    st.subheader("🎚️ Threshold & Preprocessing")
    conf_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.01,
        help="Mengatur ambang skor minimum secara real-time tanpa memuat ulang model.",
    )
    yolo_iou = st.slider(
        "NMS IoU Threshold (YOLOv26l-seg)",
        min_value=0.10,
        max_value=0.95,
        value=0.70,
        step=0.05,
    )
    yolo_imgsz = st.radio(
        "Resolusi Input YOLOv26l-seg",
        [448, 640],
        index=0,
        horizontal=True,
        help="448 = ukuran latih (proto 112x112), 640 = prototype mask lebih tajam (160x160).",
    )
    use_clahe = st.checkbox(
        "Aktifkan Domain Alignment (Grayscale + CLAHE)",
        value=False,
        help="Menyelaraskan foto RGB smartphone ke distribusi kontras grayscale CRT dataset latih.",
    )

    st.divider()
    st.subheader("🎨 Visualisasi Overlay")
    mask_alpha = st.slider("Opasitas Mask", min_value=0.10, max_value=0.90, value=0.45, step=0.05)
    show_boxes = st.checkbox("Tampilkan Bounding Box", value=True)
    show_contours = st.checkbox("Tampilkan Kontur Mask", value=True)
    show_labels = st.checkbox("Tampilkan Label & Skor", value=True)


# ======================== MAIN TABS ========================
tab1, tab2, tab3, tab4 = st.tabs([
    "🖼️ Inferensi Langsung",
    "📊 Tampilan AP Satu Kelas",
    "⚖️ Perbandingan Multi-Model",
    "📈 Benchmark & Log Training",
])

# ------------------------------------------------------------------
# TAB 1: LIVE INFERENCE + SINGLE-CLASS INSPECTOR
# ------------------------------------------------------------------
with tab1:
    img_rgb, img_name, img_hash, gts, is_manual_gt = load_image_and_gt_ui("tab1")

    if img_rgb is None:
        st.info("Silakan unggah foto ruangan indoor atau pilih salah satu foto sampel di atas untuk memulai segmentasi langsung.")
    else:
        with st.spinner(f"Menjalankan inferensi {selected_model} ({selected_scenario})..."):
            all_dets, elapsed_ms, proc_img = run_cached_inference(
                img_rgb, img_hash, selected_model, selected_scenario, yolo_imgsz, yolo_iou, use_clahe
            )

        filtered_dets = [d for d in all_dets if d.score >= conf_thresh]
        detected_class_names = sorted({d.class_name for d in filtered_dets})

        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            selected_filter_classes = st.multiselect(
                "Filter Kelas Tampil (kosongkan untuk menampilkan semua kelas terdeteksi)",
                options=detected_class_names,
                default=[],
                key="tab1_class_filter",
            )
        with col_f2:
            eval_iou_thresh = st.slider(
                "IoU Threshold Evaluasi (jika GT ada)",
                min_value=0.30,
                max_value=0.90,
                value=0.50,
                step=0.05,
                key="tab1_eval_iou",
            )

        active_filter = selected_filter_classes if selected_filter_classes else None
        vis_dets = [d for d in filtered_dets if active_filter is None or d.class_name in active_filter]
        overlay_img = render_overlay(
            proc_img if use_clahe else img_rgb,
            vis_dets,
            alpha=mask_alpha,
            show_boxes=show_boxes,
            show_contours=show_contours,
            show_labels=show_labels,
        )

        # KPI Cards
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Objek Terdeteksi", f"{len(vis_dets)}", f"dari {len(all_dets)} kandidat (>=0.05)")
        k2.metric("Kelas Unik", f"{len({d.class_name for d in vis_dets})}")
        k3.metric("Waktu Inferensi", f"{elapsed_ms:.0f} ms", get_device().upper())
        mean_sc = float(np.mean([d.score for d in vis_dets])) if vis_dets else 0.0
        k4.metric("Rata-rata Confidence", f"{mean_sc:.3f}", f"Thr >= {conf_thresh:.2f}")

        # Image Columns
        if gts:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(proc_img if use_clahe else img_rgb, caption=f"Input: {img_name} ({img_rgb.shape[1]}x{img_rgb.shape[0]})", width="stretch")
            with c2:
                st.image(overlay_img, caption=f"Prediksi {selected_model} (Skn {selected_scenario}, conf>={conf_thresh:.2f})", width="stretch")
            with c3:
                gt_vis = render_gt_overlay(img_rgb, gts)
                gt_tag = "ADE20K Manual" if is_manual_gt else "COCO-Panoptic 74"
                st.image(gt_vis, caption=f"Ground Truth ({gt_tag}, {len(gts)} objek)", width="stretch")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.image(proc_img if use_clahe else img_rgb, caption=f"Input: {img_name} ({img_rgb.shape[1]}x{img_rgb.shape[0]})", width="stretch")
            with c2:
                st.image(overlay_img, caption=f"Prediksi {selected_model} (Skn {selected_scenario}, conf>={conf_thresh:.2f})", width="stretch")

        # Downloads
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "⬇️ Unduh Gambar Hasil Overlay (.png)",
                data=encode_png_bytes(overlay_img),
                file_name=f"{Path(img_name).stem}_{selected_model}_overlay.png",
                mime="image/png",
            )
        with dl2:
            export_json = json.dumps(
                [
                    {
                        "category_id": d.category_id,
                        "class_name": d.class_name,
                        "score": round(d.score, 4),
                        "bbox_xyxy": list(d.bbox),
                        "area_px": d.area_px,
                        "area_pct": round(d.area_pct, 2),
                    }
                    for d in vis_dets
                ],
                indent=2,
            )
            st.download_button(
                "⬇️ Unduh Hasil Deteksi (.json)",
                data=export_json,
                file_name=f"{Path(img_name).stem}_{selected_model}_detections.json",
                mime="application/json",
            )

        st.divider()
        st.subheader("🔍 Inspektor Satu Kelas & Evaluasi Single-Image")
        ap_data = get_per_class_ap_cached()
        all_orig_cls_names = sorted(set(get_orig_names().values()))

        # Build selectable single-class options
        if is_manual_gt:
            inspect_options = sorted(
                {ORIG_TO_CANON.get(d.class_name, d.class_name) for d in filtered_dets}
                | {MANUAL_TO_CANON.get(g.class_name, g.class_name) for g in gts}
            )
            if not inspect_options:
                inspect_options = [c for c in CANONICAL_GROUPS if c != "other"]
            st.caption("Mode GT Manual (ADE20K -> Grup Kanonik) aktif agar kelas GT & prediksi dapat dicocokkan langsung.")
        else:
            gt_cls_set = {g.class_name for g in gts}
            inspect_options = sorted(set(detected_class_names) | gt_cls_set)
            if not inspect_options:
                inspect_options = all_orig_cls_names

        if inspect_options:
            target_single_cls = st.selectbox(
                "Pilih Satu Kelas untuk Diisolasi & Dilihat Metrik AP-nya",
                options=inspect_options,
                index=0,
                key="tab1_single_cls_select",
            )

            col_s1, col_s2 = st.columns([1.2, 1])
            with col_s1:
                iso_img = render_single_class_isolation(
                    proc_img if use_clahe else img_rgb,
                    filtered_dets,
                    target_single_cls,
                    alpha=min(0.85, mask_alpha + 0.15),
                    gts=gts if gts else None,
                    use_canonical=is_manual_gt,
                    canon_map_pred=ORIG_TO_CANON,
                    canon_map_gt=MANUAL_TO_CANON,
                )
                cap_iso = f"Isolasi Mask Kelas '{target_single_cls}' (Hijau = Prediksi" + (", Merah = Ground Truth)" if gts else ")")
                st.image(iso_img, caption=cap_iso, width="stretch")

            with col_s2:
                matched_preds = [
                    d for d in filtered_dets
                    if (ORIG_TO_CANON.get(d.class_name, d.class_name) if is_manual_gt else d.class_name) == target_single_cls
                ]
                tot_area = sum(d.area_px for d in matched_preds)
                tot_pct = sum(d.area_pct for d in matched_preds)
                max_sc = max((d.score for d in matched_preds), default=0.0)
                st.markdown(f"**Statistik Prediksi Kelas `{target_single_cls}` pada Foto Ini**")
                m_a, m_b, m_c = st.columns(3)
                m_a.metric("Jumlah Prediksi", f"{len(matched_preds)}")
                m_b.metric("Skor Tertinggi", f"{max_sc:.3f}")
                m_c.metric("Cakupan Area", f"{tot_pct:.2f}%", f"{tot_area:,} px")

                if gts:
                    sc_metrics = compute_single_image_class_metrics(
                        filtered_dets, gts, target_single_cls, iou_thresh=eval_iou_thresh, use_canonical=is_manual_gt
                    )
                    st.markdown(f"**Evaluasi Single-Image Kelas `{target_single_cls}` (IoU >= {eval_iou_thresh:.2f})**")
                    g1, g2, g3 = st.columns(3)
                    g1.metric("Single-Img AP", f"{sc_metrics['ap']:.4f}" if sc_metrics["ap"] is not None else "N/A")
                    g2.metric("Precision / Recall", f"{sc_metrics['precision']:.2f} / {sc_metrics['recall']:.2f}")
                    g3.metric("Mean IoU (GT)", f"{sc_metrics['mean_iou']:.4f}", f"TP={sc_metrics['tp']} FP={sc_metrics['fp']} FN={sc_metrics['fn']}")

                if ap_data and not is_manual_gt:
                    cls_entry = next((v for v in ap_data["classes"].values() if v["name"] == target_single_cls), None)
                    if cls_entry and selected_model in cls_entry["metrics"]:
                        bm = cls_entry["metrics"][selected_model]
                        st.markdown(f"**Benchmark Test Set Asli Kelas `{target_single_cls}` ({selected_model} Best)**")
                        b1, b2, b3 = st.columns(3)
                        b1.metric("Test AP@50", f"{bm['ap50']:.4f}" if bm["ap50"] is not None else "N/A (0 GT)")
                        b2.metric("Test AP@50-95", f"{bm['ap50_95']:.4f}" if bm["ap50_95"] is not None else "N/A")
                        b3.metric("GT / Pred Test", f"{cls_entry['gt_count']} / {bm['pred_count_25']}")

        # Detection Table
        st.subheader("📋 Daftar Objek Terdeteksi")
        if vis_dets:
            det_df = pd.DataFrame(
                [
                    {
                        "No": idx + 1,
                        "Kelas": d.class_name,
                        "ID Kelas": d.category_id,
                        "Confidence": round(d.score, 4),
                        "BBox (x1, y1, x2, y2)": f"{d.bbox}",
                        "Area (px)": d.area_px,
                        "Area (%)": round(d.area_pct, 2),
                    }
                    for idx, d in enumerate(vis_dets)
                ]
            )
            st.dataframe(det_df, width="stretch", hide_index=True)

            cls_summary = Counter(d.class_name for d in vis_dets)
            sum_df = pd.DataFrame([{"Kelas": k, "Jumlah Objek": v} for k, v in cls_summary.most_common()])
            fig_cnt = px.bar(sum_df, x="Kelas", y="Jumlah Objek", title="Distribusi Jumlah Deteksi per Kelas pada Foto Ini", text_auto=True)
            fig_cnt.update_layout(height=300, margin=dict(t=40, b=20))
            st.plotly_chart(fig_cnt, width="stretch")
        else:
            st.warning("Tidak ada objek yang melewati Confidence Threshold saat ini. Coba turunkan threshold atau aktifkan Grayscale + CLAHE.")


# ------------------------------------------------------------------
# TAB 2: PER-CLASS AP EXPLORER
# ------------------------------------------------------------------
with tab2:
    st.subheader("📊 Tampilan AP Satu Kelas & Kurva Precision-Recall (Test Set Asli)")
    ap_data = get_per_class_ap_cached()
    if ap_data is None:
        st.error("File `assets/per_class_ap.json` belum tersedia. Jalankan `python scripts/compute_per_class_ap.py`.")
    else:
        meta = ap_data["metadata"]
        st.caption(
            f"Evaluasi `pycocotools.COCOeval` pada **{meta['num_images']} gambar** Test Set Asli "
            f"({meta['num_total_annotations']} anotasi, **{meta['num_active_gt_classes']} kelas aktif** dengan GT > 0 dari {meta['num_classes_total']} kelas)."
        )

        # Overall Macro KPIs across the 3 best models
        m_cols = st.columns(3)
        for col_w, mname in zip(m_cols, ["Mask2Former-Large", "YOLOv26l-seg", "Mask-RCNN-Large"]):
            ms = ap_data["models"][mname]
            col_w.metric(
                f"{mname} (Skn {ms['scenario']})",
                f"mAP@50: {ms['mAP50']:.4f}",
                f"mAP@50-95: {ms['mAP50_95']:.4f} | mAP@75: {ms['mAP75']:.4f}",
            )

        st.divider()
        col_c1, col_c2 = st.columns([2, 1])
        with col_c2:
            only_active = st.checkbox("Hanya tampilkan kelas dengan GT > 0", value=True, key="tab2_only_active")
        classes_list = list(ap_data["classes"].values())
        if only_active:
            classes_list = [c for c in classes_list if c["gt_count"] > 0]
        classes_list.sort(key=lambda c: (-c["gt_count"], c["name"]))

        with col_c1:
            cls_labels = [f"{c['name']} (ID {c['id']} | GT={c['gt_count']})" for c in classes_list]
            chosen_cls_label = st.selectbox("Pilih Satu Kelas untuk Inspeksi AP & Kurva PR", cls_labels, index=0, key="tab2_cls_sel")
            chosen_cls = classes_list[cls_labels.index(chosen_cls_label)]

        st.markdown(f"### Detail Performa Kelas: **`{chosen_cls['name']}`** (ID `{chosen_cls['id']}`, GT Test Set = `{chosen_cls['gt_count']}`)")
        model_names_order = ["Mask2Former-Large", "YOLOv26l-seg", "Mask-RCNN-Large"]
        row_cards = st.columns(3)
        for wcol, mname in zip(row_cards, model_names_order):
            cm = chosen_cls["metrics"].get(mname, {})
            ap50_v = f"{cm.get('ap50'):.4f}" if cm.get("ap50") is not None else "N/A"
            ap75_v = f"{cm.get('ap75'):.4f}" if cm.get("ap75") is not None else "N/A"
            ap_all_v = f"{cm.get('ap50_95'):.4f}" if cm.get("ap50_95") is not None else "N/A"
            ar_v = f"{cm.get('ar100'):.4f}" if cm.get("ar100") is not None else "N/A"
            with wcol:
                st.info(
                    f"**{mname}**\n\n"
                    f"- **AP@50**: `{ap50_v}`\n"
                    f"- **AP@75**: `{ap75_v}`\n"
                    f"- **AP@50-95**: `{ap_all_v}`\n"
                    f"- **AR@100**: `{ar_v}`\n"
                    f"- **Prediksi (>=0.25 / >=0.05)**: `{cm.get('pred_count_25', 0)} / {cm.get('pred_count_05', 0)}`"
                )

        # Charts side by side: PR Curve & Single-Class Bar Comparison
        ch1, ch2 = st.columns([1.3, 1])
        with ch1:
            recall_grid = meta["recall_grid"]
            fig_pr = go.Figure()
            palette = {"Mask2Former-Large": "#2B6CB0", "YOLOv26l-seg": "#2F855A", "Mask-RCNN-Large": "#DD6B20"}
            has_pr_data = False
            for mname in model_names_order:
                pr_vals = chosen_cls["metrics"].get(mname, {}).get("pr_curve_50", [])
                if pr_vals and any(v is not None for v in pr_vals):
                    has_pr_data = True
                    fig_pr.add_trace(
                        go.Scatter(
                            x=recall_grid,
                            y=[v if v is not None else 0.0 for v in pr_vals],
                            mode="lines",
                            name=f"{mname} (AP@50={chosen_cls['metrics'][mname]['ap50']:.3f})",
                            line=dict(width=3, color=palette[mname]),
                        )
                    )
            if has_pr_data:
                fig_pr.update_layout(
                    title=f"Kurva Precision-Recall (IoU=0.50) — Kelas '{chosen_cls['name']}'",
                    xaxis_title="Recall",
                    yaxis_title="Precision",
                    xaxis=dict(range=[0, 1]),
                    yaxis=dict(range=[0, 1.05]),
                    height=380,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                )
                st.plotly_chart(fig_pr, width="stretch")
            else:
                st.warning("Kelas ini memiliki 0 anotasi Ground Truth pada subset 60 gambar Test Set Asli sehingga kurva PR tidak terdefinisi.")

        with ch2:
            bar_rows = []
            for mname in model_names_order:
                cm = chosen_cls["metrics"].get(mname, {})
                bar_rows.append({"Model": mname, "Metrik": "AP@50", "Skor": cm.get("ap50") or 0.0})
                bar_rows.append({"Model": mname, "Metrik": "AP@75", "Skor": cm.get("ap75") or 0.0})
                bar_rows.append({"Model": mname, "Metrik": "AP@50-95", "Skor": cm.get("ap50_95") or 0.0})
                bar_rows.append({"Model": mname, "Metrik": "AR@100", "Skor": cm.get("ar100") or 0.0})
            df_bar = pd.DataFrame(bar_rows)
            fig_bar = px.bar(
                df_bar,
                x="Metrik",
                y="Skor",
                color="Model",
                barmode="group",
                text_auto=".3f",
                color_discrete_map=palette,
                title=f"Komparasi Metrik Kelas '{chosen_cls['name']}'",
            )
            fig_bar.update_layout(height=380, yaxis=dict(range=[0, 1.08]))
            st.plotly_chart(fig_bar, width="stretch")

        st.divider()
        st.subheader("🏆 Peringkat AP Seluruh Kelas Aktif")
        rank_col1, rank_col2 = st.columns(2)
        with rank_col1:
            rank_model = st.selectbox("Urutkan Berdasarkan Model", model_names_order, index=0, key="tab2_rank_m")
        with rank_col2:
            rank_metric = st.selectbox("Metrik Peringkat", ["ap50", "ap50_95", "ap75", "ar100", "gt_count"], index=0, key="tab2_rank_k")

        table_rows = []
        for c in ap_data["classes"].values():
            if c["gt_count"] == 0:
                continue
            row = {"ID": c["id"], "Kelas": c["name"], "GT Count": c["gt_count"]}
            for mname in model_names_order:
                cm = c["metrics"].get(mname, {})
                short = mname.split("-")[0]
                row[f"{short} AP50"] = cm.get("ap50")
                row[f"{short} AP50-95"] = cm.get("ap50_95")
                row[f"{short} Pred(0.25)"] = cm.get("pred_count_25", 0)
            row["_sort_key"] = (
                c["gt_count"] if rank_metric == "gt_count" else (c["metrics"].get(rank_model, {}).get(rank_metric) or -1.0)
            )
            table_rows.append(row)

        df_rank = pd.DataFrame(table_rows).sort_values("_sort_key", ascending=False).drop(columns=["_sort_key"])
        st.dataframe(df_rank, width="stretch", hide_index=True)

        top15 = df_rank.head(15)
        col_target = f"{rank_model.split('-')[0]} AP50" if rank_metric != "ap50_95" else f"{rank_model.split('-')[0]} AP50-95"
        if col_target in top15.columns:
            fig_top = px.bar(
                top15.sort_values(col_target, ascending=True),
                x=col_target,
                y="Kelas",
                orientation="h",
                text_auto=".3f",
                title=f"Top-15 Kelas Terbaik pada {rank_model} ({col_target})",
            )
            fig_top.update_layout(height=450)
            st.plotly_chart(fig_top, width="stretch")


# ------------------------------------------------------------------
# TAB 3: SIDE-BY-SIDE MULTI-MODEL COMPARISON
# ------------------------------------------------------------------
with tab3:
    st.subheader("⚖️ Perbandingan Multi-Model pada Foto yang Sama")
    st.caption(
        "Memakai Confidence Threshold, YOLO Resolution, dan toggle Grayscale + CLAHE dari Sidebar. "
        "Anda dapat membandingkan YOLOv26l-seg, Mask2Former-Large, dan Mask-RCNN-Large secara berdampingan."
    )
    img3, name3, hash3, gts3, is_man3 = load_image_and_gt_ui("tab3")
    models_to_compare = st.multiselect(
        "Pilih Model untuk Dibandingkan (gunakan skenario terbaik masing-masing atau skenario sidebar)",
        list(MODEL_REGISTRY.keys()),
        default=["Mask2Former-Large", "YOLOv26l-seg", "Mask-RCNN-Large"],
        key="tab3_models",
    )
    scen_mode = st.radio(
        "Pilihan Skenario Checkpoint",
        ["Gunakan Skenario Terbaik per Model (M2F=B, YOLO=B, MRCNN=C)", f"Gunakan Skenario Sidebar ({selected_scenario})"],
        horizontal=True,
        key="tab3_scen_mode",
    )

    if img3 is not None and models_to_compare:
        if st.button("🚀 Jalankan Perbandingan Semua Model Terpilih", type="primary", key="tab3_run_btn"):
            cols = st.columns(len(models_to_compare))
            summary_rows = []
            for col_u, mname in zip(cols, models_to_compare):
                scen_use = str(MODEL_REGISTRY[mname]["default_scenario"]) if "Terbaik" in scen_mode else selected_scenario
                with col_u:
                    with st.spinner(f"Memproses {mname} ({scen_use})..."):
                        raw_dets, ms_t, proc3 = run_cached_inference(
                            img3, hash3, mname, scen_use, yolo_imgsz, yolo_iou, use_clahe
                        )
                    f_dets = [d for d in raw_dets if d.score >= conf_thresh]
                    ov3 = render_overlay(
                        proc3 if use_clahe else img3,
                        f_dets,
                        alpha=mask_alpha,
                        show_boxes=show_boxes,
                        show_contours=show_contours,
                        show_labels=show_labels,
                    )
                    st.image(ov3, caption=f"{mname} (Skn {scen_use}) — {len(f_dets)} det | {ms_t:.0f} ms", width="stretch")
                    top_cls = ", ".join([f"{k} ({v})" for k, v in Counter(d.class_name for d in f_dets).most_common(5)])
                    st.caption(f"Kelas terbanyak: {top_cls or '-'}")
                    summary_rows.append({
                        "Model": mname,
                        "Skenario": scen_use,
                        "Deteksi (>= thr)": len(f_dets),
                        "Kelas Unik": len({d.class_name for d in f_dets}),
                        "Rata-rata Skor": round(float(np.mean([d.score for d in f_dets])), 4) if f_dets else 0.0,
                        "Waktu (ms)": round(ms_t, 1),
                    })
            st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)
    elif img3 is None:
        st.info("Pilih foto sampel atau upload foto di atas, lalu tekan tombol Jalankan Perbandingan.")


# ------------------------------------------------------------------
# TAB 4: BENCHMARK, AUGMENTASI & LOG TRAINING
# ------------------------------------------------------------------
with tab4:
    st.subheader("📈 Hasil Eksperimen Augmentasi, Benchmark GPU & Log Training Asli")
    df_bench = get_benchmark_df()
    if not df_bench.empty:
        c_b1, c_b2 = st.columns(2)
        with c_b1:
            fig_m50 = px.bar(
                df_bench,
                x="model",
                y="mAP50",
                color="scenario",
                barmode="group",
                text_auto=".3f",
                title="Validation mAP@50: Skenario Augmentasi A vs B vs C",
            )
            st.plotly_chart(fig_m50, width="stretch")
        with c_b2:
            fig_vram = px.bar(
                df_bench,
                x="model",
                y="wall_clock_time_sec",
                color="scenario",
                barmode="group",
                text_auto=".0f",
                title="Durasi Komputasi GPU 2x NVIDIA T4 (Detik)",
            )
            st.plotly_chart(fig_vram, width="stretch")
        st.dataframe(df_bench, width="stretch", hide_index=True)

    st.divider()
    st.subheader("📉 Kurva Log Training Asli per Epoch (`results.csv`)")
    lc1, lc2 = st.columns(2)
    with lc1:
        log_model = st.selectbox("Model Log", list(MODEL_REGISTRY.keys()), index=0, key="tab4_log_m")
    with lc2:
        log_scen = st.selectbox("Skenario Log", ["A", "B", "C"], index=1, key="tab4_log_s")

    df_log = load_training_log(log_model, log_scen)
    if not df_log.empty:
        ep_col = "epoch" if "epoch" in df_log.columns else df_log.columns[0]
        m50_col = "mAP50" if "mAP50" in df_log.columns else "metrics/mAP50(M)"
        m5095_col = "mAP50-95" if "mAP50-95" in df_log.columns else "metrics/mAP50-95(M)"
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(x=df_log[ep_col], y=df_log[m50_col], mode="lines+markers", name="Validation mAP@50"))
        if m5095_col in df_log.columns:
            fig_curve.add_trace(go.Scatter(x=df_log[ep_col], y=df_log[m5095_col], mode="lines+markers", name="Validation mAP@50-95"))
        fig_curve.update_layout(title=f"Kurva Validasi {log_model} — Skenario {log_scen}", xaxis_title="Epoch", yaxis_title="Mask mAP", height=360)
        st.plotly_chart(fig_curve, width="stretch")
        with st.expander("Lihat Tabel Log `results.csv` Lengkap"):
            st.dataframe(df_log, width="stretch", hide_index=True)

    st.divider()
    st.subheader("📦 Status Ekspor ONNX & TensorFlow Lite")
    tflite_rep = get_tflite_report()
    if tflite_rep:
        st.dataframe(pd.DataFrame(tflite_rep), width="stretch", hide_index=True)
