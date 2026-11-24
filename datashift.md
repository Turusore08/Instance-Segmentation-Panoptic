# Analisis Domain Shift: Test Set Panoptic Indoor vs Dataset Manual

Dokumen ini membandingkan secara kuantitatif perbedaan metrik evaluasi (*instance segmentation*, `pycocotools.COCOeval`) antara **Test Set Asli Panoptic Indoor** (`panoptic-indoor-segmentation.v1i.yolov8-obb/test`) dan **Dataset Manual** (`viskom-data-sendiri` dengan *ground truth* 74-kelas dari `sota_annotations.coco.json` menggunakan pemetaan kelas **1-to-1 exact match** `sota_category_id = model_category_id + 1`).

Seluruh evaluasi dijalankan pada tiga bobot **Large** terbaik hasil pelatihan 15 epoch pada `2x NVIDIA T4` (`IMG_SIZE = 448`):
- **Mask2Former-Large** (`Skenario B` — Default Framework)
- **YOLOv26l-seg** (`Skenario B` — Default Framework)
- **Mask-RCNN-Large** (`Skenario C` — Custom Albumentations Indoor, `ResNet-101-FPN`)

---

## 1. Ringkasan Eksekutif & Profil Kedua Test Set

| Karakteristik | Test Set Asli (`Panoptic Indoor`) | Dataset Manual (`viskom-data-sendiri`) |
| :--- | :--- | :--- |
| **Jumlah Gambar Uji** | `60` sampel evaluasi lintas-model (dari total `359` gambar test / `5.344` anotasi; split asli `1.191` train / `166` valid / `359` test) | `40` foto penuh kamera smartphone (`140` anotasi) |
| **Resolusi & Aspek Rasio** | Seragam `432 x 432` piksel (**1:1 square stretch**) | Didominasi *portrait* **9:16** (`720x1280`: 35 foto, `899x1599`: 3 foto, `960x1280`: 1 foto, `2250x4000`: 1 foto) |
| **Ruang Warna** | **Grayscale CRT** + Auto-contrast (R = G = B identik pada 50/50 sampel) | **True RGB** berwarna penuh (`40/40` foto) |
| **Bentuk Ground Truth** | Anotasi kotak miring **OBB 4-titik** | **Poligon organik presisi piksel** (`Mask2Former Swin-L + Grounding DINO Base + SAM 2.1 Large`) |
| **Taksonomi Kelas** | `74` kelas (`48` kelas aktif dengan GT > 0 pada subset 60 gambar) | `74` kelas identik (`37` kelas aktif dengan GT > 0; `27` kelas beririsan langsung dengan subset 60 gambar asli) |
| **Distribusi Skala Area GT** | Dominan kecil–menengah akibat kanvas `432x432` | `2` *small* (`< 32^2` px), `13` *medium* (`32^2–96^2` px), `125` *large* (`>= 96^2` px) |

**Temuan Utama**:
- **`Mask2Former-Large`** paling tahan terhadap pergeseran domain: `mAP@50` hanya turun **`-7,6%`** (`0.2757` -> `0.2547`), bahkan **`mAP@75` naik `+10,8%`** (`0.1429` -> `0.1584`) dan **`mAP@50-95` naik `+1,2%`** (`0.1526` -> `0.1544`) karena masker *cross-attention* lebih selaras dengan kontur poligon organik daripada OBB 4-titik.
- **`YOLOv26l-seg`** mengalami penurunan *domain shift* terparah: `mAP@50` anjlok **`-56,0%`** (`0.2044` -> `0.0899`) dan `mAR@100` turun **`-70,7%`** (`0.2392` -> `0.0702`), dengan jumlah prediksi `conf >= 0.25` menyusut drastis (`63` deteksi pada 40 foto).
- **`Mask-RCNN-Large`** turun **`-44,7%`** pada `mAP@50` (`0.1091` -> `0.0603`), namun evaluasi *class-agnostic* membuktikan bahwa **86% kegagalan berasal dari kepala klasifikasi**, bukan lokalisasi mask (`Class-Agnostic mAP@50 = 0.4429`).

---

## 2. Perbandingan Utama Metrik Evaluasi (74-Kelas Exact Match)

| Model | Metrik | Test Set Asli (60 img) | Dataset Manual RGB (40 img) | Selisih Absolut ($\Delta$) | Domain Shift (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Mask2Former-Large** | `mAP@50` | `0.2757` | `0.2547` | `-0.0210` | **`-7.6%`** |
| **Mask2Former-Large** | `mAP@75` | `0.1429` | `0.1584` | `+0.0155` | **`+10.8%`** |
| **Mask2Former-Large** | `mAP@50-95` | `0.1526` | `0.1544` | `+0.0018` | **`+1.2%`** |
| **Mask2Former-Large** | `mAR@100` | `0.2564` | `0.2050` | `-0.0514` | `-20.0%` |
| **YOLOv26l-seg** | `mAP@50` | `0.2044` *(Full 359: `0.1502`)* | `0.0899` | `-0.1145` | **`-56.0%`** |
| **YOLOv26l-seg** | `mAP@75` | `0.1140` | `0.0548` | `-0.0592` | `-51.9%` |
| **YOLOv26l-seg** | `mAP@50-95` | `0.1184` *(Full 359: `0.0793`)* | `0.0492` | `-0.0692` | **`-58.4%`** |
| **YOLOv26l-seg** | `mAR@100` | `0.2392` | `0.0702` | `-0.1690` | **`-70.7%`** |
| **Mask-RCNN-Large** | `mAP@50` | `0.1091` | `0.0603` | `-0.0488` | **`-44.7%`** |
| **Mask-RCNN-Large** | `mAP@75` | `0.0464` | `0.0305` | `-0.0159` | `-34.3%` |
| **Mask-RCNN-Large** | `mAP@50-95` | `0.0524` | `0.0293` | `-0.0231` | `-44.1%` |
| **Mask-RCNN-Large** | `mAR@100` | `0.1319` | `0.0651` | `-0.0668` | `-50.6%` |

> **Catatan Referensi Validasi Training Asli** (`166` gambar `valid`): `Mask2Former-Large (B)` `0.3552 / 0.2092`, `YOLOv26l-seg (B)` `0.3065 / 0.1882`, `Mask-RCNN-Large (C)` `0.1976 / 0.0988`. Selisih dari `valid` ke `manual RGB` lebih besar lagi karena gambar `valid` masih berbagi distribusi pencahayaan grayscale CRT dengan `train`.

---

## 3. Breakdown Performa Berdasarkan Skala Objek (`AP_small`, `AP_medium`, `AP_large`)

| Model | Test Set Asli (`Small / Medium / Large`) | Dataset Manual RGB (`Small / Medium / Large`) | Perubahan Signifikan |
| :--- | :---: | :---: | :--- |
| **Mask2Former-Large** | `0.0460 / 0.1550 / 0.2260` | `0.0000 / 0.0936 / 0.1615` | Objek kecil (`2` GT di manual, mis. benda kecil di meja `720x1280`) tidak tertangkap akibat kompresi resolusi ke `IMG_SIZE=448`. |
| **YOLOv26l-seg** | `0.0440 / 0.1620 / 0.1490` | `0.0000 / 0.0667 / 0.0573` | Penurunan merata pada skala *medium* (`-58,8%`) maupun *large* (`-61,5%`) karena *stride-4 prototype mask* dan konvolusi awal kehilangan respons pada foto RGB *portrait*. |
| **Mask-RCNN-Large** | `0.0040 / 0.0640 / 0.0900` | `0.0000 / 0.0223 / 0.0310` | Performa *small* tetap `0.0000` (bottleneck grid RoIAlign `28x28` saat input diubah ke `448x448`), sedangkan skala *medium* & *large* turun ~`65%`. |

---

## 4. Dekomposisi Domain Shift: Lokalisasi vs Klasifikasi (`Class-Aware` vs `Class-Agnostic`)

Untuk mengetahui apakah model gagal **menemukan objek (lokalisasi/segmentasi)** atau **menebak nama kelas (klasifikasi)** di dataset manual, semua label GT dan prediksi digabung menjadi satu kelas tunggal (`object`) pada evaluasi *class-agnostic*.

| Model | Manual `Class-Aware` (`mAP@50 / 50-95 / mAR@100`) | Manual `Class-Agnostic` (`mAP@50 / 50-95 / mAR@100`) | Rasio Lonjakan `mAP@50` | Interpretasi |
| :--- | :---: | :---: | :---: | :--- |
| **Mask2Former-Large** | `0.2547 / 0.1544 / 0.2050` | **`0.4721 / 0.2893 / 0.4700`** | **1,85x** (`+85,4%`) | Hampir separuh mask objek sudah akurat secara spasial, namun sebagian tertukar ke kelas *stuff/thing* mirip. |
| **YOLOv26l-seg** | `0.0899 / 0.0492 / 0.0702` | **`0.4131 / 0.2167 / 0.3707`** | **4,60x** (`+359,5%`) | Lokalisasi cukup baik (`0.4131`), namun `head` klasifikasi gagal memberi label yang tepat pada 74 kelas dalam domain RGB. |
| **Mask-RCNN-Large** | `0.0603 / 0.0293 / 0.0651` | **`0.4429 / 0.2229 / 0.4400`** | **7,35x** (`+634,5%`) | RPN dan `mask_predictor` melokalisasi `44,3%` objek dengan baik, tetapi `FastRCNNPredictor` melabeli ulang objek dengan kelas dominan data latih. |

**Kesimpulan Dekomposisi**: Pada ketiga arsitektur, **selisih terbesar disebabkan oleh *classification/semantic shift***, bukan kebutaan geometris total. Model masih mengenali batas fisik benda di foto smartphone (`mAP@50 Class-Agnostic = 0.4131–0.4721`), tetapi skor logit kelas melenceng ke subkelas bertetangga (mis. `dining table` vs `table-merged`, `floor-wood` vs `floor-other-merged`, `paper` vs `paper-merged`).

---

## 5. Uji Domain Shift Fotometrik: `RGB Asli` vs `Grayscale + CLAHE` pada Dataset Manual

Untuk menguji seberapa besar pengaruh perubahan ruang warna (**Grayscale CRT** pada data latih vs **RGB** pada foto smartphone), seluruh 40 foto manual dikonversi ke `Grayscale + CLAHE (clipLimit=2.0, tileGridSize=8x8)` lalu dievaluasi ulang dengan GT 74-kelas yang sama.

| Model | Kondisi Input | `mAP@50` | `mAP@75` | `mAP@50-95` | `mAR@100` | Prediksi (`conf >= 0.25 / >= 0.05`) | Efek `Grayscale + CLAHE` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **YOLOv26l-seg** | RGB Asli | `0.0899` | `0.0548` | `0.0492` | `0.0702` | `63 / 195` | Baseline manual |
| **YOLOv26l-seg** | **Grayscale + CLAHE** | **`0.1068`** | `0.0428` | **`0.0532`** | `0.0700` | **`81 / 217`** | **`mAP@50` naik `+18,8%`**, prediksi `>=0.25` naik `+28,6%` |
| **Mask2Former-Large** | RGB Asli | **`0.2547`** | **`0.1584`** | **`0.1544`** | **`0.2050`** | `160 / 299` | Baseline manual terbaik |
| **Mask2Former-Large** | Grayscale + CLAHE | `0.2113` | `0.1032` | `0.1086` | `0.1618` | `162 / 319` | `mAP@50` turun `-17,0%` |
| **Mask-RCNN-Large** | RGB Asli | **`0.0603`** | **`0.0305`** | **`0.0293`** | `0.0651` | `266 / 805` | Baseline manual |
| **Mask-RCNN-Large** | Grayscale + CLAHE | `0.0491` | `0.0254` | `0.0264` | **`0.0704`** | `266 / 796` | `mAP@50` turun `-18,6%` |

**Analisis Temuan**:
- **Arsitektur CNN Single-Stage (`YOLOv26l-seg`)** sensitif terhadap pergeseran warna dan kontras karena dilatih dari bobot `yolo26l-seg.pt` lalu beradaptasi kuat pada citra grayscale CRT. Menyamakan input manual ke *Grayscale + CLAHE* langsung memulihkan `mAP@50` sebesar **`+18,8%`** (`0.0899 -> 0.1068`).
- **Sebaliknya**, `Mask2Former-Large` (dengan *differential LR* `0.1x` menjaga representasi *pretraining* COCO RGB `Swin-Large`) justru **dirugikan jika kanal warna dibuang** (`0.2547 -> 0.2113`), karena fitur kromatik membantu membedakan kelas bertekstur mirip seperti `wall-brick`, `book`, dan `rug-merged`.

---

## 6. Perbandingan Domain Shift Per-Kelas (`AP@50`: Test Set Asli vs Dataset Manual RGB)

Tabel berikut memuat seluruh `37` kelas aktif di Dataset Manual beserta perbandingan `AP@50` terhadap Test Set Asli (`60` gambar). Kelas dengan `GT Asli = 0` pada subset 60 gambar diberi tanda `-` (tetap dihitung dalam mAP Dataset Manual jika muncul pada `40` foto manual).

| Kelas | GT Asli | GT Manual | Mask2Former (Asli -> Manual) | YOLOv26l-seg (Asli -> Manual) | Mask-RCNN (Asli -> Manual) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `paper` | `0` | `15` | `-` -> `0.0000` | `-` -> `0.0000` | `-` -> `0.0347` |
| `floor-other-merged` | `37` | `14` | `0.3919` -> `0.2003` | `0.3385` -> `0.1273` | `0.3138` -> `0.0392` |
| `wall-other-merged` | `108` | `11` | `0.2363` -> `0.2127` | `0.2459` -> **`0.4158`** | `0.2421` -> `0.1733` |
| `cup` | `10` | `10` | `0.3465` -> `0.1749` | `0.0693` -> **`0.2079`** | `0.0000` -> `0.0000` |
| `bottle` | `4` | `7` | `0.2052` -> `0.1485` | `0.5692` -> `0.2871` | `0.0000` -> `0.0000` |
| `scissors` | `0` | `7` | `-` -> `0.0000` | `-` -> `0.0000` | `-` -> `0.0000` |
| `person` | `66` | `6` | `0.6455` -> **`0.5633`** | `0.5581` -> `0.3527` | `0.3212` -> `0.2525` |
| `table-merged` | `9` | `6` | `0.2544` -> `0.0659` | `0.1204` -> `0.0962` | `0.1206` -> `0.0321` |
| `chair` | `25` | `6` | `0.6807` -> `0.0000` | `0.6950` -> `0.0000` | `0.1542` -> `0.0000` |
| `dining table` | `7` | `5` | `0.0000` -> `0.0000` | `0.0000` -> `0.0000` | `0.0000` -> `0.0000` |
| `box` | `0` | `5` | `-` -> `0.0000` | `-` -> `0.0000` | `-` -> `0.0231` |
| `book` | `9` | `4` | `0.3729` -> **`0.5050`** | `0.3793` -> `0.0000` | `0.0031` -> `0.0000` |
| `rug-merged` | `10` | `4` | `0.1040` -> **`0.2574`** | `0.0000` -> `0.0000` | `0.0545` -> `0.0000` |
| `ceiling-merged` | `27` | `3` | `0.6174` -> `0.0000` | `0.2071` -> `0.0000` | `0.3807` -> `0.0000` |
| `laptop` | `0` | `3` | `-` -> **`0.5545`** | `-` -> `0.0000` | `-` -> `0.0000` |
| `floor-wood` | `13` | `3` | `0.7272` -> `0.0000` | `0.2664` -> `0.0000` | `0.2430` -> `0.0000` |
| `cell phone` | `1` | `3` | `0.0000` -> **`0.3366`** | `0.0000` -> `0.0000` | `0.0000` -> `0.0000` |
| `light` | `37` | `2` | `0.0865` -> `0.0000` | `0.0745` -> `0.0000` | `0.0264` -> `0.0000` |
| `wall-brick` | `5` | `2` | `0.0208` -> **`0.7525`** | `0.0000` -> `0.0000` | `0.0247` -> `0.0000` |
| `window-other` | `12` | `2` | `0.2528` -> `0.0000` | `0.0736` -> `0.0000` | `0.1294` -> `0.0842` |
| `wallet` | `5` | `2` | `0.7171` -> **`1.0000`** | `0.8020` -> `0.5050` | `0.6832` -> **`1.0000`** |
| `paper-merged` | `0` | `2` | `-` -> `0.0000` | `-` -> `0.0000` | `-` -> `0.0000` |
| `keyboard` | `0` | `2` | `-` -> `0.0000` | `-` -> `0.0000` | `-` -> `0.0000` |
| `trash-bin` | `1` | `2` | `0.0000` -> `0.4000` | `0.1667` -> **`0.8350`** | `0.0000` -> `0.0631` |
| `bowl` | `4` | `2` | `0.3317` -> `0.2525` | `0.1881` -> `0.0000` | `0.0728` -> `0.0000` |
| `pillow` | `21` | `1` | `0.2937` -> `0.0000` | `0.2338` -> `0.0000` | `0.2372` -> `0.0000` |
| `backpack` | `13` | `1` | `0.8247` -> **`1.0000`** | `0.1741` -> `0.0000` | `0.0198` -> `0.0000` |
| `handbag` | `116` | `1` | `0.2383` -> **`1.0000`** | `0.2651` -> `0.0000` | `0.0017` -> `0.0000` |
| `sink` | `15` | `1` | `0.2694` -> **`1.0000`** | `0.2252` -> `0.0000` | `0.0000` -> `0.0000` |
| `clock` | `0` | `1` | `-` -> `0.0000` | `-` -> `0.0000` | `-` -> `0.0000` |
| `tree-merged` | `0` | `1` | `-` -> `0.0000` | `-` -> `0.0000` | `-` -> `0.0000` |
| `plug` | `1` | `1` | `0.0000` -> **`1.0000`** | `0.0000` -> `0.5000` | `0.0000` -> `0.5000` |
| `mouse` | `0` | `1` | `-` -> `0.0000` | `-` -> `0.0000` | `-` -> `0.0000` |
| `wall-wood` | `5` | `1` | `0.0000` -> `0.0000` | `0.0000` -> `0.0000` | `0.0000` -> `0.0000` |
| `counter` | `17` | `1` | `0.0864` -> `0.0000` | `0.0353` -> `0.0000` | `0.0074` -> `0.0294` |
| `spoon` | `1` | `1` | `0.0000` -> `0.0000` | `0.0909` -> `0.0000` | `0.0000` -> `0.0000` |
| `cardboard` | `0` | `1` | `-` -> `0.0000` | `-` -> `0.0000` | `-` -> `0.0000` |

**Observasi Kritis Per-Kelas**:
1. **Kelas Paling Stabil / Membaik di Manual**: Pada `Mask2Former-Large`, kelas `person` (`0.6455 -> 0.5633`), `book` (`0.3729 -> 0.5050`), `wallet` (`0.7171 -> 1.0000`), `backpack` (`0.8247 -> 1.0000`), `laptop` (`0.5545`), `wall-brick` (`0.0208 -> 0.7525`), dan `sink` (`0.2694 -> 1.0000`) tetap tinggi karena memiliki siluet khas dan dukungan *pretraining* COCO.
2. **Kelas dengan Drop Total ke `0.0000`**:
   - **`chair` (`6` GT)**, **`ceiling-merged` (`3` GT)**, dan **`floor-wood` (`3` GT)** turun dari `0.62–0.73` menjadi `0.0000` di ketiga model akibat perbedaan sudut kamera *close-up/portrait*, material lantai kayu/ubin lokal, dan kebingungan antar-subkelas (*fine-grained stuff split* seperti `floor-wood` diprediksi sebagai `floor-other-merged`).
   - **Objek meja & alat tulis kecil (`paper` 15 GT, `scissors` 7 GT, `box` 5 GT)**: menyumbang `27` dari `140` anotasi manual (`19,3%`), namun skor hampir seluruhnya `0.0000` karena jumlah latihnya minim di *Panoptic Indoor* serta ketiadaan fitur skala besar untuk benda kecil pada `IMG_SIZE=448`.

---

## 7. Sintesis Penyebab Domain Shift & Rekomendasi Mitigasi

1. **Ketidakcocokan Domain Fotometrik (*Grayscale CRT* vs *RGB Kamera*)**: Paling memukul `YOLOv26l-seg` (`-56,0%` `mAP@50`). Solusi: sertakan *random grayscale / color-jitter* kuat saat pelatihan atau terapkan *CLAHE grayscale alignment* khusus saat inferensi YOLO.
2. **Kompresi Aspek Rasio (*1:1 432x432* vs *9:16 Portrait*)**: Memaksa foto `720x1280` menjadi `448x448` merusak proporsi vertikal objek seperti `chair`, `table-merged`, dan `person`. Solusi: latih dengan *letterbox padding* bervariasi dan resolusi lebih tinggi (`imgsz >= 640`).
3. **Granularitas Kelas *Stuff* yang Ambigu (`74` Kelas Terpecah)**: Subkelas yang secara visual mirip (`floor-wood` vs `floor-other-merged`, `paper` vs `paper-merged`, `dining table` vs `table-merged`) menciptakan *gap* ekstrem antara `Class-Agnostic` (`0.41–0.47`) dan `Class-Aware` (`0.06–0.25`). Solusi: lakukan penggabungan label semantik (*taxonomy consolidation*) atau *fine-tuning* ringan dengan anotasi poligon organik SOTA.
4. **Geometri Anotasi Latih (OBB 4-Titik vs Poligon Organik)**: Kotak OBB melatih regressor/mask head dengan toleransi latar tinggi. `Mask2Former-Large` terbukti paling kebal (`mAP@50-95` bahkan naik ke `0.1544`) karena struktur *query mask-attention* mempertahankan kontur batas yang bersih.

---

*Artefak data dan skrip reproduksi*: [`scripts/evaluate_datashift.py`](scripts/evaluate_datashift.py), [`assets/per_class_ap.json`](assets/per_class_ap.json), [`assets/datashift_metrics.json`](assets/datashift_metrics.json).
