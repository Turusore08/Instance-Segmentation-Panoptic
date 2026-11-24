# Experiment Logbook: Instance Segmentation Benchmark (Kaggle 2x T4)

- **Dataset**: `/kaggle/input/datasets/melekbengi/data-viskom/data viskom` (`74` classes)
- **Hardware**: `2x GPU` (`[0, 1]`), `IMG_SIZE=448`, `EPOCHS=15`, `FP16 AMP=True`

## 1. Tabel Perbandingan Metrik & Benchmark GPU

| Model | Skenario Augmentasi | mAP50 | mAP50-95 | Wall-Clock Time (s) | Total Peak VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Mask2Former-Large | A | 0.3661 | 0.2066 | 7808.34 | 11154.00 |
| Mask2Former-Large | B | 0.3552 | 0.2092 | 7818.18 | 12100.00 |
| Mask2Former-Large | C | 0.3624 | 0.2045 | 7966.51 | 12044.00 |
| Mask-RCNN-Large | A | 0.1887 | 0.0966 | 2948.77 | 4234.00 |
| Mask-RCNN-Large | B | 0.1940 | 0.0970 | 2937.82 | 4318.00 |
| Mask-RCNN-Large | C | 0.1976 | 0.0988 | 3005.60 | 4400.00 |
| YOLOv26l-seg (yolo26l-seg) | A | 0.2831 | 0.1632 | 1099.78 | 10686.00 |
| YOLOv26l-seg (yolo26l-seg) | B | 0.3065 | 0.1882 | 1144.39 | 11756.00 |
| YOLOv26l-seg (yolo26l-seg) | C | 0.3021 | 0.1771 | 1160.27 | 9998.00 |

## 2. Kendala Teknis (Technical Issues & Resolutions)

| Tanggal / Fase | Model / Skenario | Kendala yang Ditemukan | Solusi / Workaround yang Diterapkan |
| :--- | :--- | :--- | :--- |
| Preprocessing | Semua Model | Format dataset awal `yolov8-obb` belum kompatibel dengan Mask2Former & Mask R-CNN, dan rasterisasi 2D `instance_map` menyebabkan 54,5% oklusi mask yang tumpang tindih | Konversi otomatis ke COCO JSON poligon/mask & YOLO-Seg YAML di `WORKING_DIR`; gunakan 3D binary `mask_labels (N, H, W)` di Mask2Former dan `overlap_mask=False` di Ultralytics |
| Setup Environment | Kaggle Python 3.13 | `pip install --upgrade` menimpa `numpy` di disk saat `numpy.ufunc` lama masih termuat di memori (`AttributeError`) | Kunci versi paket inti (`numpy`, `scipy`, `torch`, `torchvision`, `tensorflow`, `albumentations`) di `pip_constraints.txt` dan instal hanya modul yang belum ada |
| Multi-GPU Training | Mask2Former-Large / Mask R-CNN-Large | `nn.parallel.replicate`/`DataParallel` gagal sinkronisasi Python list (`RuntimeError: tensors on different devices`), `expandable_segments:True` + FP16 deformable attention `grid_sample` memicu `CUDA illegal memory access`, serta `/dev/shm` Worker Aborted | Ganti dengan explicit `copy.deepcopy` replicas pada GPU 0 & 1 (`BATCH_SIZE=2`, `GRAD_ACCUM_STEPS=4`, `NUM_WORKERS=0`, `pin_memory=False`), sinkronisasi gradien & parameter manual, serta subclass `FP32Mask2FormerPixelDecoderEncoder` |
| Optimasi Mask2Former | Mask2Former-Large | Konvergensi kepala klasifikasi/query lambat saat learning rate seragam dan evaluasi maskless/class-probability terputus | Terapkan differential learning rate (`backbone 0.1x`, `head 1.0x`), `CosineAnnealingLR`, dan maskness scoring (`cls_prob * mean_fg_prob`) saat evaluasi |
| Export TFLite | Mask2Former-Large / Mask-RCNN-Large / YOLOv26l-seg | `onnx2tf` gagal dengan `ModuleNotFoundError: No module named 'ai_edge_litert'`, sedangkan `YOLOv26l-seg` `quantize=16 (FP16)` ditolak LiteRT | Simpan `.onnx` FP32 untuk Mask2Former-Large (869 MB) & Mask-RCNN-Large (243 MB), dan ekspor `yolo26l_seg_best_int8.tflite` (29 MB) via kuantisasi INT8 |

## 3. Keputusan Arsitektural & Analisis Augmentasi

- **Perbandingan Skenario A vs B vs C**:
  - *Observasi Skenario A (Raw / Resize + Normalisasi)*: Pada `Mask2Former-Large`, Skenario A mencapai `mAP50=0.3661` (`mAP50-95=0.2066`), namun `val_loss` mencapai titik terbaik di epoch 7 (`42.8714`) lalu terus naik hingga `47.4237` di epoch 15 sementara `train_loss` turun tajam (`64.1871` -> `17.8991`), menandakan overfitting lokal paling cepat. Pada `YOLOv26l-seg`, Skenario A hanya mencapai `mAP50=0.2831` (`mAP50-95=0.1632`) dengan `val/seg_loss` terbaik `2.2633` (epoch 5) yang naik ke `2.6404` di epoch 15.
  - *Observasi Skenario B (Default Framework)*: Memberikan peningkatan signifikan pada `YOLOv26l-seg` (`mAP50=0.3065`, `mAP50-95=0.1882`, `val/seg_loss=2.0552`) berkat Mosaic, HSV, dan scale jittering bawaan Ultralytics. `Mask2Former-Large` Skenario B menghasilkan `mAP50-95` validasi tertinggi (`0.2092`, `mAP50=0.3552`, `best_val_loss=41.9611`), sedangkan `Mask-RCNN-Large` naik tipis ke `mAP50=0.1940` (`mAP50-95=0.0970`).
  - *Observasi Skenario C (Custom Albumentations Indoor)*: `RandomBrightnessContrast`, `HorizontalFlip`, `ShiftScaleRotate`, dan `GaussianBlur` secara konsisten menekan `val_loss` hingga level terendah pada `Mask2Former-Large` (`41.3677` di epoch 7, `mAP50=0.3624`, `mAP50-95=0.2045`) dan `YOLOv26l-seg` (`2.0034` di epoch 15, `mAP50=0.3021`), serta memberikan skor tertinggi bagi `Mask-RCNN-Large` (`mAP50=0.1976`, `mAP50-95=0.0988`).
- **Pemilihan Model Terbaik untuk Deployment**:
  - *Akurasi vs Latensi vs Ukuran `.tflite`*: Untuk akurasi dan ketahanan lintas domain (*server/cloud inference*), **`Mask2Former-Large` (Skenario B)** adalah model terbaik (`val mAP50-95 = 0.2092`; `test manual class-aware mAP50 = 0.2829` & `mAP50-95 = 0.1943`), mengungguli `YOLOv26l-seg` dan `Mask-RCNN-Large`, meski berbiaya komputasi tertinggi (~7.818 detik/run, 12.100 MB VRAM, ONNX 869 MB). Untuk *edge/mobile deployment*, **`YOLOv26l-seg` (Skenario B)** paling praktis karena waktu latih tercepat (`1.144,39` detik), `mAP50=0.3065` pada validasi (melampaui baseline Roboflow `yolov11n-seg` `0.254`), dan berhasil diekspor ke `yolo26l_seg_best_int8.tflite` berukuran `29 MB`.

## 4. Lokasi Artefak Output

- Benchmark JSON: `/kaggle/working/gpu_benchmark.json`
- Benchmark CSV: `/kaggle/working/gpu_benchmark.csv`
- Logs & TensorBoard: `/kaggle/working/logs`
- Model Checkpoints: `/kaggle/working/checkpoints`
- TFLite Exports: `/kaggle/working/exports_tflite`
