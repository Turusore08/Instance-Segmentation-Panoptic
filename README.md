# Indoor Instance Segmentation — Laporan Analisis Proyek & Data Shift (Model Large)

Proyek ini membandingkan kinerja tiga arsitektur *instance segmentation* kapasitas **Large** (`Mask2Former-Large`, `YOLOv26l-seg`, dan `Mask-RCNN-Large`) pada **74 kelas ruangan indoor** (`Panoptic Indoor Segmentation`), dengan fokus utama menganalisis **data shift (pergeseran distribusi data)** ketika model yang dilatih pada dataset sintetis/terkurasi (**Panoptic Indoor**) dihadapkan pada foto kamera *smartphone* dunia nyata (**Dataset Manual** `viskom-data-sendiri`).

> Laporan rincian per-kelas (`37` kelas aktif) tersedia di [`datashift.md`](datashift.md), sementara seluruh metrik mentah dapat diaudit di [`assets/per_class_ap.json`](assets/per_class_ap.json) dan [`assets/datashift_metrics.json`](assets/datashift_metrics.json).

---

## 1. Ringkasan Proyek & Baseline Pelatihan (`2x NVIDIA T4`)

Pelatihan dilakukan di lingkungan Kaggle Notebook (`2x NVIDIA T4`, `IMG_SIZE = 448`, `15` epoch, *FP16 Mixed Precision*) melintasi **3 skenario augmentasi** (`A`: Tanpa Augmentasi/Raw, `B`: Default Framework, `C`: Custom Albumentations Indoor) dengan total waktu komputasi **35.889,66 detik (~9,97 jam)** untuk 9 *run*.

| Model Terbaik (Dari 9 Run) | Skenario Terbaik | Val `mAP@50` | Val `mAP@50-95` | Durasi Latih (s) | Peak VRAM Total (MB) | Catatan |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Mask2Former-Large** (`Swin-Large`) | **B** *(A tertinggi mAP@50)* | `0.3552` *(A: `0.3661`)* | **`0.2092`** | `7.818,18` | `12.100` | Melonjak ~7x dari varian Base (`0.0536`) berkat *differential LR* (`0.1x` backbone) & 3D binary masks. |
| **YOLOv26l-seg** (`yolo26l-seg`) | **B** | `0.3065` | `0.1882` | `1.144,39` | `11.756` | Tercepat (~19 menit), mengungguli baseline Roboflow (`yolov11n-seg` `mAP@50 = 0.2540`), ekspor TFLite INT8 (`29 MB`) sukses. |
| **Mask-RCNN-Large** (`ResNet-101-FPN`) | **C** | `0.1976` | `0.0988` | `3.005,60` | `4.400` | Paling hemat VRAM, namun konvergensi kepala klasifikasi belum optimal dalam 15 epoch. |

---

## 2. Profil Sumber Data Shift (`Panoptic Indoor` vs `Dataset Manual`)

Evaluasi *domain shift* diuji secara adil dengan pemetaan **1-to-1 exact match pada 74 kelas identik** (`sota_category_id = model_category_id + 1`), di mana ground truth dataset manual dihasilkan dari fusi `Mask2Former Swin-Large + Grounding DINO Base + SAM 2.1 Large` (`140` poligon organik).

| Dimensi Pergeseran | Test Set Asli (`Panoptic Indoor`) | Dataset Manual (`viskom-data-sendiri`) | Dampak Teknis ke Model |
| :--- | :--- | :--- | :--- |
| **Jumlah Sampel & Anotasi** | `60` sampel evaluasi lintas-model *(subset dari `359` gambar test / `5.344` anotasi; split: `1.191` train / `166` valid)* | `40` foto penuh kamera smartphone (`140` anotasi) | Dataset manual merepresentasikan kondisi *zero-shot deployment* dunia nyata. |
| **Ruang Warna (*Photometric Shift*)** | **Grayscale CRT** + auto-contrast (kanal `R = G = B` identik pada 50/50 sampel) | **True RGB** berwarna penuh (`40/40` foto) | Ekstraktor fitur awal CNN yang beradaptasi pada gradien monokromatik terganggu oleh saturasi warna RGB. |
| **Resolusi & Rasio Aspek (*Geometric Shift*)** | Seragam `432 x 432` piksel (**1:1 square stretch**) | **9:16 portrait** (`720x1280`: 35 foto, `899x1599`: 3 foto, `960x1280`: 1 foto, `2250x4000`: 1 foto) | Kompresi foto portrait tinggi ke kanvas persegi `448x448` mengubah proporsi furnitur vertikal (`chair`, `table-merged`, `person`). |
| **Geometri Ground Truth** | Kotak miring **OBB 4-titik** kasar | **Poligon organik presisi piksel** | Mask yang bersih pada IoU tinggi (`mAP@75`) menuntut kontur segmentasi presisi, bukan sekadar bounding box kasar. |
| **Taksonomi & Skala Objek** | `74` kelas (`48` kelas aktif di subset 60 gambar) | `74` kelas identik (`37` kelas aktif; `27` beririsan dengan subset asli; skala: `2` small, `13` medium, `125` large) | `10` kelas aktif di manual (`paper` 15 GT, `scissors` 7 GT, `box` 5 GT, dll.) memiliki `0` sampel di subset 60 gambar test asli. |

---

## 3. Analisis Kuantitatif Data Shift

### 3.1 Perbandingan Metrik Utama (`Test Set Asli` vs `Dataset Manual RGB 74-Kelas`)

| Model Terbaik | Metrik | Test Set Asli (60 img) | Dataset Manual RGB (40 img) | Selisih ($\Delta$) | Domain Shift (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Mask2Former-Large** | `mAP@50` | `0.2757` | `0.2547` | `-0.0210` | **`-7,6%`** |
| **Mask2Former-Large** | `mAP@75` | `0.1429` | `0.1584` | `+0.0155` | **`+10,8%`** |
| **Mask2Former-Large** | `mAP@50-95` | `0.1526` | `0.1544` | `+0.0018` | **`+1,2%`** |
| **Mask2Former-Large** | `mAR@100` | `0.2564` | `0.2050` | `-0.0514` | `-20,0%` |
| **YOLOv26l-seg** | `mAP@50` | `0.2044` *(Full 359: `0.1502`)* | `0.0899` | `-0.1145` | **`-56,0%`** |
| **YOLOv26l-seg** | `mAP@75` | `0.1140` | `0.0548` | `-0.0592` | `-51,9%` |
| **YOLOv26l-seg** | `mAP@50-95` | `0.1184` *(Full 359: `0.0793`)* | `0.0492` | `-0.0692` | **`-58,4%`** |
| **YOLOv26l-seg** | `mAR@100` | `0.2392` | `0.0702` | `-0.1690` | **`-70,7%`** |
| **Mask-RCNN-Large** | `mAP@50` | `0.1091` | `0.0603` | `-0.0488` | **`-44,7%`** |
| **Mask-RCNN-Large** | `mAP@75` | `0.0464` | `0.0305` | `-0.0159` | `-34,3%` |
| **Mask-RCNN-Large** | `mAP@50-95` | `0.0524` | `0.0293` | `-0.0231` | `-44,1%` |
| **Mask-RCNN-Large** | `mAR@100` | `0.1319` | `0.0651` | `-0.0668` | `-50,6%` |

**Interpretasi Utama**:
- **`Mask2Former-Large` paling invarian terhadap data shift**: `mAP@50` hanya menyusut `-7,6%`, bahkan **`mAP@75` melonjak `+10,8%`** (`0.1429 -> 0.1584`) dan **`mAP@50-95` naik `+1,2%`** (`0.1526 -> 0.1544`). Mekanisme *masked cross-attention* menghasilkan mask halus yang lebih cocok dievaluasi pada poligon organik manual dibanding anotasi kotak OBB kasar di test set asli.
- **`YOLOv26l-seg` paling rentan terhadap pergeseran domain**: penurunan `mAP@50` hingga **`-56,0%`** dan `mAR@100` **`-70,7%`**, ditandai turunnya prediksi aktif (`conf >= 0.25`) menjadi hanya `63` deteksi di seluruh `40` foto manual.

---

### 3.2 Dekomposisi Kegagalan: Lokalisasi vs Klasifikasi (*Class-Aware* vs *Class-Agnostic*)

Untuk memisahkan apakah model gagal **menemukan objek** atau **menebak label kelas yang benar**, dilakukan evaluasi *class-agnostic* (seluruh 74 kelas disatukan ke satu kelas `object`) pada dataset manual:

| Model | Manual *Class-Aware* (`mAP@50 / 50-95 / mAR@100`) | Manual *Class-Agnostic* (`mAP@50 / 50-95 / mAR@100`) | Rasio Lonjakan `mAP@50` | Akar Masalah Dominan |
| :--- | :---: | :---: | :---: | :--- |
| **Mask2Former-Large** | `0.2547 / 0.1544 / 0.2050` | **`0.4721 / 0.2893 / 0.4700`** | **1,85x** (`+85,4%`) | Separuh mask spasial tepat, sebagian tertukar ke kelas *stuff* mirip. |
| **YOLOv26l-seg** | `0.0899 / 0.0492 / 0.0702` | **`0.4131 / 0.2167 / 0.3707`** | **4,60x** (`+359,5%`) | Lokalisasi/masking tetap hidup (`0.4131`), namun kepala klasifikasi kehilangan kalibrasi logit di domain RGB. |
| **Mask-RCNN-Large** | `0.0603 / 0.0293 / 0.0651` | **`0.4429 / 0.2229 / 0.4400`** | **7,35x** (`+634,5%`) | **86% kegagalan berasal dari `FastRCNNPredictor`**, bukan RPN atau `mask_predictor` (yang mendeteksi `44,3%` objek). |

---

### 3.3 Uji Eksperimental Shift Fotometrik (`RGB Asli` vs `Grayscale + CLAHE`)

Seluruh `40` foto manual diubah menjadi *Grayscale + CLAHE (`clipLimit=2.0, tileGridSize=8x8`)* untuk menyelaraskan distribusi warna ke dataset latih:

| Model | Kondisi Input | `mAP@50` | `mAP@50-95` | Prediksi (`conf >= 0.25 / >= 0.05`) | Efek `Grayscale + CLAHE` |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **YOLOv26l-seg** | RGB Asli | `0.0899` | `0.0492` | `63 / 195` | Baseline manual |
| **YOLOv26l-seg** | **Grayscale + CLAHE** | **`0.1068`** | **`0.0532`** | **`81 / 217`** | **Membaik `+18,8%` `mAP@50`** & **`+28,6%` jumlah deteksi** |
| **Mask2Former-Large** | RGB Asli | **`0.2547`** | **`0.1544`** | `160 / 299` | Warna RGB membantu disambiguasi kelas bertekstur |
| **Mask2Former-Large** | Grayscale + CLAHE | `0.2113` | `0.1086` | `162 / 319` | Turun **`-17,0%`** ketika informasi warna dibuang |
| **Mask-RCNN-Large** | RGB Asli | **`0.0603`** | **`0.0293`** | `266 / 805` | Baseline manual |
| **Mask-RCNN-Large** | Grayscale + CLAHE | `0.0491` | `0.0264` | `266 / 796` | Turun **`-18,6%`** ketika informasi warna dibuang |

> **Temuan Kunci**: Penyelarasan *Grayscale + CLAHE* terbukti memulihkan **CNN single-stage (`YOLOv26l-seg`)** yang ketergantungan pada kontras tepi monokromatik data latih, tetapi **merugikan arsitektur berbasis *backbone* pretraining RGB seperti `Mask2Former-Large`** (`Swin-Large` menjaga fitur warna via *differential LR* `0.1x`).

---

### 3.4 Ringkasan Skala Objek (`AP_small / medium / large`) & Pergeseran Per-Kelas

- **Skala Objek (`Test Set Asli` → `Dataset Manual RGB`)**:
  - `Mask2Former-Large`: `0.0460 / 0.1550 / 0.2260` → `0.0000 / 0.0936 / 0.1615`
  - `YOLOv26l-seg`: `0.0440 / 0.1620 / 0.1490` → `0.0000 / 0.0667 / 0.0573`
  - `Mask-RCNN-Large`: `0.0040 / 0.0640 / 0.0900` → `0.0000 / 0.0223 / 0.0310`
  - Pada input `IMG_SIZE=448`, seluruh `2` objek *small* di foto smartphone tidak tertangkap (`0.0000`), sementara degradasi utama terjadi pada objek *medium* dan *large* milik `YOLO` serta `Mask R-CNN`.
- **Pola Per-Kelas (`37` kelas aktif di Manual, rincian lengkap di [`datashift.md`](datashift.md))**:
  - **Stabil / Tinggi di Kedua Domain**: `person` (`Mask2Former`: `0.6455 -> 0.5633`), `book` (`0.3729 -> 0.5050`), `wallet` (`0.7171 -> 1.0000`), `backpack` (`0.8247 -> 1.0000`), `sink` (`0.2694 -> 1.0000`), `laptop` (`0.5545`), `wall-brick` (`0.0208 -> 0.7525`), serta `wall-other-merged` pada YOLO (`0.2459 -> 0.4158`).
  - **Kolaps ke `0.0000` Akibat Domain Shift**: `chair` (`6` GT, turun dari `0.68–0.70` di asli ke `0.0000`), `ceiling-merged` (`3` GT, dari `0.21–0.62` ke `0.0000`), `floor-wood` (`3` GT, dari `0.24–0.73` ke `0.0000`, terserap menjadi `floor-other-merged`), dan benda meja kecil seperti `paper` (`15` GT), `scissors` (`7` GT), serta `box` (`5` GT).

---

## 4. Bukti Visual: 12 Panel Perbandingan Segmentasi (`Test Asli` vs `Dataset Manual`)

Seluruh 12 panel perbandingan dirender pada `conf >= 0.25` dengan urutan kolom: **Ground Truth | Mask2Former-Large (Skn B) | YOLOv26l-seg (Skn B) | Mask-RCNN-Large (Skn C)**.

### Sampel Test Set Asli (`432x432` Grayscale CRT)
- **`orig_01.png`** (`312_jpg...jpg`, `16` GT) — Mask2Former `29 det | mIoU 0.519`, YOLO `12 det | mIoU 0.321`, Mask R-CNN `21 det | mIoU 0.459`:
  ![orig_01](assets/segmentation_comparisons/orig_01.png)
- **`orig_03.png`** (`341_jpg...jpg`, `13` GT) — Mask2Former `17 det | mIoU 0.661`, YOLO `12 det | mIoU 0.550`, Mask R-CNN `21 det | mIoU 0.553`:
  ![orig_03](assets/segmentation_comparisons/orig_03.png)

### Sampel Dataset Manual (`viskom-data-sendiri` RGB Portrait + GT 74-Kelas SOTA)
- **`manual_01.png`** (`WhatsApp Image 2026-11-23 at 00.08.33 (2).jpeg`, `18` GT) — Mask2Former `29 det | mIoU 0.416`, YOLO `10 det | mIoU 0.275`, Mask R-CNN `26 det | mIoU 0.292` (memperlihatkan fragmentasi & salah kelas seperti `stove-merged` di Mask R-CNN):
  ![manual_01](assets/segmentation_comparisons/manual_01.png)
- **`manual_06.png`** (`WhatsApp Image 2026-11-23 at 00.09.14 (1).jpeg`, `5` GT) — Mask2Former `4 det | mIoU 0.530`, **YOLOv26l-seg gagal total `0 det | mIoU 0.000`**, Mask R-CNN `12 det | mIoU 0.665`:
  ![manual_06](assets/segmentation_comparisons/manual_06.png)

<details>
<summary><b>Lihat 8 Panel Perbandingan Lainnya (<code>orig_02</code>, <code>orig_04–06</code>, <code>manual_02–05</code>) & Ringkasan Kuantitatif Lengkap</b></summary>

| File Panel | Foto Asal | GT | Mask2Former-Large (Det \| mIoU) | YOLOv26l-seg (Det \| mIoU) | Mask-RCNN-Large (Det \| mIoU) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| [`orig_02.png`](assets/segmentation_comparisons/orig_02.png) | `299_jpg...jpg` | 16 | `24` \| `0.5080` | `10` \| `0.3129` | `17` \| `0.4720` |
| [`orig_04.png`](assets/segmentation_comparisons/orig_04.png) | `339_jpg...jpg` | 15 | `30` \| `0.6144` | `10` \| `0.3359` | `19` \| `0.3977` |
| [`orig_05.png`](assets/segmentation_comparisons/orig_05.png) | `329_jpg...jpg` | 14 | `17` \| `0.5951` | `5` \| `0.2733` | `20` \| `0.4230` |
| [`orig_06.png`](assets/segmentation_comparisons/orig_06.png) | `291_jpg...jpg` | 12 | `24` \| `0.5852` | `15` \| `0.5118` | `17` \| `0.3700` |
| [`manual_02.png`](assets/segmentation_comparisons/manual_02.png) | `WhatsApp Image ... 00.09.06.jpeg` | 8 | `6` \| `0.4796` | `1` \| `0.2210` | `18` \| `0.5270` |
| [`manual_03.png`](assets/segmentation_comparisons/manual_03.png) | `WhatsApp Image ... 00.09.14.jpeg` | 6 | `5` \| `0.4860` | `1` \| `0.1857` | `8` \| `0.6483` |
| [`manual_04.png`](assets/segmentation_comparisons/manual_04.png) | `WhatsApp Image ... 00.09.17 (1).jpeg` | 6 | `9` \| `0.5817` | `2` \| `0.3277` | `20` \| `0.6404` |
| [`manual_05.png`](assets/segmentation_comparisons/manual_05.png) | `WhatsApp Image ... 00.08.33.jpeg` | 5 | `2` \| `0.9411` | `2` \| `0.9471` | `3` \| `0.8886` |

- ![orig_02](assets/segmentation_comparisons/orig_02.png)
- ![orig_04](assets/segmentation_comparisons/orig_04.png)
- ![orig_05](assets/segmentation_comparisons/orig_05.png)
- ![orig_06](assets/segmentation_comparisons/orig_06.png)
- ![manual_02](assets/segmentation_comparisons/manual_02.png)
- ![manual_03](assets/segmentation_comparisons/manual_03.png)
- ![manual_04](assets/segmentation_comparisons/manual_04.png)
- ![manual_05](assets/segmentation_comparisons/manual_05.png)

</details>

---

## 5. Rekomendasi Mitigasi Data Shift

1. **Penyelarasan Domain Inferensi Selektif Arsitektur**: Aktifkan *Grayscale + CLAHE* secara bawaan hanya saat menjalankan `YOLOv26l-seg` (`+18,8%` `mAP@50`), sedangkan `Mask2Former-Large` dan `Mask-RCNN-Large` harus tetap menerima input RGB penuh.
2. **Augmentasi Pelatihan Multi-Domain**: Tambahkan *Strong ColorJitter / RandomGrayscale* pada data latih grayscale serta *letterbox multi-aspect-ratio padding* agar model tidak overfit terhadap bingkai `432x432` persegi.
3. **Konsolidasi Subkelas *Stuff* yang Tumpang Tindih**: Gabungkan subkelas bertetangga (`floor-wood` + `floor-other-merged`, `paper` + `paper-merged`, `table-merged` + `dining table`) atau lakukan *domain-adaptive fine-tuning* memakai poligon organik hasil auto-labeling SOTA.
4. **Peningkatan Resolusi Inferensi**: Naikkan `imgsz = 640` (prototype mask `160x160`) untuk memulihkan deteksi objek kecil seperti `paper`, `scissors`, dan `box`.

---

## 6. Struktur Repositori & Cara Menjalankan

```text
Instance-Segmentation-Panoptic/
├── app.py                                   # Aplikasi Web Streamlit (Inferensi, AP Satu Kelas, Multi-Model, Benchmark)
├── datashift.md                             # Laporan terperinci analisis Domain Shift 74-kelas
├── train.ipynb                              # Notebook pelatihan 3 arsitektur Large x 3 skenario di Kaggle 2x T4
├── mask2former_instance_segmentation_kaggle.ipynb
├── requirements.txt
├── src/
│   ├── config.py                            # Registri checkpoint, 74 kelas, dan utilitas sampel
│   ├── inference.py                         # Loader model, Grayscale+CLAHE alignment, & renderer overlay
│   └── metrics.py                           # Kalkulator single-image per-class AP/IoU & pemuat log
├── scripts/
│   ├── compute_per_class_ap.py              # Generator assets/per_class_ap.json (COCOeval 60 img asli)
│   ├── evaluate_datashift.py                # Evaluator domain shift 74-kelas & class-agnostic -> datashift_metrics.json
│   └── generate_12_comparisons.py           # Generator 12 panel PNG perbandingan di assets/segmentation_comparisons/
└── assets/
    ├── per_class_ap.json
    ├── datashift_metrics.json
    └── segmentation_comparisons/            # 12 file PNG panel perbandingan + summary.json
```

### Menjalankan Aplikasi Web Streamlit Secara Lokal

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/streamlit run app.py --server.port 8501
```

### Reproduksi Evaluasi Data Shift & 12 Panel Gambar

```bash
.venv/bin/python scripts/evaluate_datashift.py
.venv/bin/python scripts/generate_12_comparisons.py
```
