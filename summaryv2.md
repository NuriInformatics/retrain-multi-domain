# Ringkasan Kritis Jurnal: Identifikasi Otomatis SATD Multi-Sumber

> **Rujukan:** Li, Y., Soliman, M., & Avgeriou, P. (2023). *Automatic identification of self-admitted technical debt from four different sources*. **Empirical Software Engineering (EMSE)**, 28(3), 65. DOI: [10.1007/s10664-023-10297-9](https://doi.org/10.1007/s10664-023-10297-9).

---

## 1. Inti Masalah & Urgensi Penelitian (*The "Why"*)

*Self-Admitted Technical Debt* (SATD) adalah utang teknis yang diakui secara sadar oleh pengembang melalui teks saat mengambil jalan pintas implementasi (*shortcuts*). Mayoritas riset terdahulu hanya fokus pada komentar kode (`TODO`/`FIXME`) atau tiket issue, sehingga mengabaikan artefak alur kerja kolaboratif modern seperti **Pesan Komit (*Commit Messages*)** dan **Tinjauan Kode (*Pull Requests*)**. Di sisi lain, pembuatan model deteksi terpisah untuk masing-masing sumber terkendala oleh minimnya data berlabel pada artefak tertentu. Untuk mengatasi kendala kelangkaan data tersebut, penulis mengusulkan **Multitask Text-CNN (MT-Text-CNN)** yang memanfaatkan *shared representation* lintas domain agar artefak yang kaya data dapat mendongkrak performa artefak yang minim data.

---

## 2. Pemisahan Research Questions (RQ1 – RQ4)

Empat pertanyaan penelitian dirancang secara berjenjang dari validasi akurasi model hingga eksplorasi empiris skala besar:

| RQ | Rumusan Pertanyaan | Rasionalitas Ilmiah | Hasil Kunci |
| :---: | :--- | :--- | :--- |
| **RQ1** | **Akurasi Model Multi-Task** | Menguji apakah transfer leksikal lintas domain mampu mengungguli model single-task dan ML klasik. | **MT-Text-CNN unggul** dengan F1 = 0.497 (scratch) dan F1 = 0.611 (optimal FastText). Peningkatan tertinggi pada dataset kecil. |
| **RQ2** | **Kata Kunci Informatif** | Menemukan karakteristik leksikon yang menjadi sinyal kuat pengakuan SATD di tiap platform. | Leksikon *Issue* dan *Pull Request* paling banyak beririsan; *Comment* memiliki kosakata paling unik dan spesifik. |
| **RQ3** | **Prevalensi Skala Industri** | Mengukur kuantitas dan sebaran tipe SATD pada proyek open source dunia nyata. | Ditemukan **1,05 juta SATD** di 103 proyek. **Code/Design Debt (`C/D.`)** mendominasi absolut (60–70%). |
| **RQ4** | **Relasi Lintas Artefak** | Melacak bagaimana utang teknis saling terhubung dan berpindah antar-artefak dalam siklus rilis. | Teridentifikasi **4 tipe relasi resmi paper**: (1) *Documenting existing SATD repeatedly*, (2) *Discussing the solution of SATD*, (3) *Documenting the repayment of SATD*, dan (4) *Repaying documentation debt in code comments*. |

---

## 3. Alur Eksperimen Lengkap dari Jurnal (*Experimental Workflow*)

Alur kerja eksperimental dirancang melalui enam tahapan terintegrasi:

```text
[1. Seleksi 103 Proyek Apache] ──> [2. Penautan Data (Link A, B, C)] ──> [3. Ground Truth (95k)]
                                                                               │
[6. Analisis Pasca-Model]      <── [5. Training Seimbang & K-Fold]   <── [4. Pra-pemrosesan Teks]
```

1. **Seleksi Subjek & Penautan Data (Bab 3.2 & 3.3):**  
   Dipilih 103 proyek Java dari ekosistem Apache dengan kriteria aktif: memiliki minimal 500 commit, 500 PR, dan 500 issue (total terkumpul 30,3 juta teks mentah). Penautan otomatis dilakukan melalui ID tiket issue pada judul PR (*Link A*), referensi PR pada pesan commit (*Link B*), dan *merged commit hash* di GitHub (*Link C*).
2. **Ground Truth & Taksonomi Label (Bab 3.5):**  
   Sebanyak 95.455 baris data berlabel dikurasi secara manual dengan reliabilitas tinggi (*Cohen's Kappa = +0.74*). Label distandarisasi ke dalam 4 tipe utang dan 1 kelas negatif:

#### Tabel 1: Distribusi Dataset Berlabel Sesuai Taksonomi Paper (Tabel 3)

| Sumber Artefak | C/D. (Code/Design) | DOC. (Documentation) | TST. (Test) | REQ. (Requirement) | Other (Non-SATD) | Total Sampel |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Code Comments** | 2.703 | 54 | 85 | 757 | 58.676 | **62.275** |
| **Commit Messages** | 522 | 98 | 58 | 27 | 4.295 | **5.000** |
| **Pull Requests** | 510 | 101 | 68 | 20 | 4.301 | **5.000** |
| **Issue Trackers** | 2.169 | 487 | 338 | 97 | 20.089 | **23.180** |
| **Total Sampel** | **5.904** | **740** | **549** | **901** | **87.361** | **95.455** |

3. **Pra-pemrosesan Teks (Bab 3.4):**  
   Menghapus delimiter komentar (`//`, `/*`), *case folding* ke huruf kecil, tokenisasi menggunakan *TweetTokenizer*, serta menyaring digit dan simbol non-semantik (namun mempertahankan tanda `!` dan `?`).
4. **Arsitektur Model & Pelatihan Seimbang (Bab 3.6.1 & 3.6.5):**  
   MT-Text-CNN menggunakan *Shared Embedding* (300-d) dan *Shared Conv2D* (filter `[3, 4, 5]` dengan 100/200 feature maps) yang terhubung ke 4 classification head linear terpisah. Pelatihan dilakukan secara stokastik seimbang (*Collobert & Weston*) berdampingan dengan *Weighted Cross-Entropy Loss* untuk menghukum kesalahan pada kelas minoritas (`REQ.` dan `TST.`).
5. **Protokol Validasi:**  
   Dievaluasi menggunakan **Stratified 10-Fold Cross-Validation** dengan metrik Macro Average F1 serta uji signifikansi statistik (*Student's T-Test* $p < 0.05$ dan *Cliff's Delta* ukuran *large*).

---

## 4. Temuan Kunci & Rekapitulasi Hasil Evaluasi

Berikut adalah perbandingan F1-score fase baseline arsitektur (Tabel 4 jurnal):

#### Tabel 2: Perbandingan Macro F1-Score Model Machine Learning vs Baseline (Tabel 4 Paper)

| Kategori | Classifier | Comment | Commit | Pull | Issue | **Overall Avg.** | Imp. over Random |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Deep Learning** | **MT-Text-CNN (Proposed)** | **0.619** | **0.475** | **0.441** | **0.453** | **0.497** | **23.7×** |
| | Single-Task Text-CNN | 0.550 | 0.451 | 0.451 | 0.431 | 0.471 | 22.4× |
| **Traditional ML** | Logistic Regression (LR) | 0.400 | 0.253 | 0.236 | 0.208 | 0.274 | 13.0× |
| | Random Forest (RF) | 0.494 | 0.238 | 0.109 | 0.083 | 0.231 | 11.0× |
| | Support Vector Machine (SVM) | 0.171 | 0.173 | 0.081 | 0.042 | 0.117 | 5.6× |
| **Baseline** | Random Classifier | 0.016 | 0.020 | 0.024 | 0.025 | 0.021 | 1.0× |

*Catatan Optimasi Lanjutan:* Pada model puncak berbasis **FastText embedding** (Bab 4.1.2), F1-score rata-rata MT-Text-CNN meningkat signifikan hingga **0.611**. Keunggulan arsitektur multi-task paling nyata pada dataset kecil (*Commit* dan *Pull Request*), membuktikan keberhasilan transfer representasi leksikal lintas sumber.

---

## 5. Karakteristik Leksikon & Relasi Antar-Sumber (RQ2, RQ3, RQ4)

Melalui penelusuran bobot konvolusi (*feature backtracking*) dan analisis relasi kualitatif (*Cohen's Kappa = +0.81*), penulis memetakan bahasa pengembang dan siklus hidup utang:

#### Tabel 3: Kata Kunci Dominan & 4 Tipe Relasi SATD Sesuai Paper (RQ2 & RQ4)

| Sumber Artefak | Kata Kunci Teratas (*Informative Keywords*) | 4 Tipe Relasi SATD Antar-Sumber (Bab 4.4 Paper) |
| :--- | :--- | :--- |
| **Comment** | `todo`, `fixme`, `hack`, `workaround`, `temporary` | **1. Documenting existing SATD repeatedly:** Mencatat utang yang ada di sumber tambahan (misal dari Issue/PR dijadikan reminder `TODO` di kode). |
| **Commit** | `workaround`, `hack`, `clean up`, `temporary fix` | **2. Discussing the solution of SATD:** Mendiskusikan solusi teknis detail utang di sumber lain (misal dilaporkan di Issue, solusi diperdebatkan di PR). |
| **Pull Request** | `future work`, `todo`, `refactor`, `tech debt` | **3. Documenting the repayment of SATD:** Mendokumentasikan pelunasan utang di sumber lain (misal commit message mencatat bahwa issue SATD telah diperbaiki). |
| **Issue** | `debt`, `refactoring`, `cleanup`, `redesign` | **4. Repaying documentation debt in code comments:** Kasus khusus di mana utang dokumentasi dilunasi langsung dengan menulis komentar kode / Javadoc. |

Pada analisis empiris terhadap 103 proyek (RQ3), ditemukan sebanyak **1.050.412 item SATD** dengan dominasi **Code/Design Debt (60–70%)**. Hasil ini membuktikan bahwa proyek modern yang berbasis alur *Pull Request* mendokumentasikan utang teknis jauh lebih banyak di *Issue* dan *Pull Request* daripada di komentar kode langsung.

---

## 6. Kesimpulan & Relevansi Akademik

1. **Pesan Utama:** Audit utang teknis tidak boleh hanya mengandalkan pencarian `TODO` pada kode program. Keputusan arsitektur dan kompromi desain krusial justru terkonsentrasi pada diskusi *Issue Trackers* dan *Pull Requests*.
2. **Keunggulan Multi-Task:** Paradigma *multi-task learning* terbukti efektif mengatasi kelangkaan data berlabel pada artefak perangkat lunak tertentu melalui pemanfaatan representasi bersama.
3. **Catatan untuk Sidang / Laporan:** Perbedaan angka F1-score antara reproduksi baseline scratch (0.431) dan skor puncak paper (0.611) adalah wajar, karena angka 0.611 dicapai pada fase optimasi lanjutan dengan FastText eksternal 300 dimensi yang dilatih pada 30 juta korpus SE, bukan model scratch acak.
