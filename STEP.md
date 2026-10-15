# Panduan Reproduksi Eksperimen SATD (Tabel 4)
**Berdasarkan Paper:**
> **Li, Y., Soliman, M., & Avgeriou, P. (2023).**  
> *Automatic identification of self-admitted technical debt from four different sources.*  
> **Empirical Software Engineering (EMSE)**, 28(3), 65.  
> DOI: [10.1007/s10664-023-10297-9](https://doi.org/10.1007/s10664-023-10297-9)

---

## 1. Latar Belakang & Tujuan Reproduksi

### Mengapa Bereproduksi dari Scratch?
Pada jurnal, penulis melaporkan dua fase eksperimen utama:
1. **Fase Baseline & Validasi Arsitektur (Tabel 4 di Bab 4.1.1):**  
   * **Inisialisasi Acak (*Randomized Word Vectors*):** Word embedding diinisialisasi secara acak via `nn.Embedding(vocab_size, 300)` murni dari data latih, **tanpa** embedding eksternal.
   * Model diuji murni menggunakan 4 dataset CSV berlabel untuk membandingkan performa arsitektur Multitask Text-CNN (MT-Text-CNN) vs Single-Task Text-CNN vs Traditional ML (LR, SVM, RF) vs Random Classifier.
   * Rata-rata F1-score yang dicapai: **~0.497**.
2. **Fase Optimasi Lanjutan (Tabel 6 s/d 9 di Bab 4.1.2):**  
   * Penulis menambahkan FastText word embedding yang dilatih pada 30+ juta teks mentah open-source (file Zenodo ~2 GB), weighted loss, dan tuning jumlah filter (200 feature maps), menghasilkan skor puncak **0.611**.

> **Tujuan Notebook `reproduce_table4.ipynb`:**  
> Mereproduksi secara mandiri (*from scratch*) seluruh matriks F1-score pada **Tabel 4** menggunakan data 4 CSV dari repositori GitHub.

---

## 2. Struktur Data & Standarisasi Label (Sesuai Tabel 3 Paper)

Data mentah terdiri dari 4 sumber file CSV di folder `./data`:
1. `satd-dataset-code_comments.csv` (62.275 baris)
2. `satd-dataset-commit_messages.csv` (5.000 baris)
3. `satd-dataset-pull_requests.csv` (5.000 baris)
4. `satd-dataset-issues.csv` (23.180 baris)

### Aturan Pemetaan Label (*Label Mapping*):
Sesuai **Bab 3.5 & Tabel 3 paper**, technical debt dikelompokkan menjadi **4 tipe utama** dan 1 kelas negatif (**non-SATD**):

| Label Mentah di CSV | Label Standar di Model | Kode Kelas | Keterangan |
| :--- | :--- | :---: | :--- |
| `code_debt`, `design_debt`, `code/design_debt` | **C/D.** (`code|design-debt`) | `1` | Penggabungan Code Debt & Design Debt |
| `documentation_debt` | **DOC.** (`documentation-debt`) | `2` | Hutang dokumentasi |
| `test_debt` | **TST.** (`test-debt`) | `3` | Hutang pengujian/unit test |
| `requirement_debt` | **REQ.** (`requirement-debt`) | `4` | Hutang fungsionalitas/persyaratan |
| `non_debt`, `defect_debt`, `architecture_debt`, `build_debt` | **Other** (`non-SATD`) | `0` | Kelas negatif / bukan 4 debt di atas |

*Tabel Distribusi Data (Cocok dengan Tabel 3 Paper):*
* **Comment:** C/D = 2.703 | DOC = 54 | TST = 85 | REQ = 757 | Other = 58.676 (Total: 62.275)
* **Commit:** C/D = 522 | DOC = 98 | TST = 58 | REQ = 27 | Other = 4.295 (Total: 5.000)
* **Pull:** C/D = 510 | DOC = 101 | TST = 68 | REQ = 20 | Other = 4.301 (Total: 5.000)
* **Issue:** C/D = 2.169 | DOC = 487 | TST = 338 | REQ = 97 | Other = 20.089 (Total: 23.180)

---

## 3. Langkah Demi Langkah Preprocessing Teks (Bab 3.4)

Fungsi `preprocess_text` pada notebook mengimplementasikan tahapan pembersihan teks persis seperti pada script resmi penulis (`satd_detector.py`):
1. **Pembersihan Sintaks Komentar:** Menghilangkan delimiter `//`, `/*`, dan `*/` dengan ekspresi reguler `r'(//)|(/\*)|(\*/)'`.
2. **Case Normalization:** Mengubah seluruh karakter ke huruf kecil (*lowercase*).
3. **Tokenisasi Kalimat & Kata:** Memisahkan kalimat dengan `nltk.sent_tokenize()` dan token kata dengan `nltk.TweetTokenizer()`.
4. **Pembersihan Simbol & Noise:**
   * Token bertanda koma `,` diubah menjadi titik `.`.
   * Menghilangkan token yang memuat karakter spasi, `#`, `_`, `/`, `>`, `<`, `@`.
   * Menghilangkan token yang memuat digit/angka.
   * Menghapus karakter tanda baca (kecuali tanda `!` dan `?` yang informatif).

---

## 4. Protokol Evaluasi: Stratified 10-Fold Cross-Validation (Bab 3.6.4)

Karena data sangat tidak seimbang (*class imbalance*), validasi silang dilakukan dengan:
* **Stratified 10-Fold:** Dataset dipecah menjadi 10 partisi, di mana setiap partisi mempertahankan proporsi kelas yang sama.
* **Rotasi:** 9 partisi digunakan untuk training dan 1 partisi untuk testing, diulang 10 kali.
* **Perhitungan Metrik per Fold:**
  Untuk tiap kelas target $c \in \{\text{C/D.}, \text{DOC.}, \text{TST.}, \text{REQ.}\}$:
  $$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$
  $$F1_c = 2 \times \frac{\text{Precision}_c \times \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$
* **AVG (Macro F1):** Rata-rata F1-score dari 4 tipe target debt:
  $$\text{AVG} = \frac{F1_{\text{C/D}} + F1_{\text{DOC}} + F1_{\text{TST}} + F1_{\text{REQ}}}{4}$$
* **Avg. (Kolom):** Rata-rata skor across 4 sumber data (Comment, Commit, Pull, Issue).

---

## 5. Implementasi Model (Bab 3.6.1 & 3.6.2)

### A. Baseline: Random Classifier
* **Mekanisme:** Menghitung probabilitas apriori kemunculan tiap kelas dari data latih $P(Y = c) = \frac{N_c}{N}$.
* **Prediksi:** Menghasilkan kelas secara acak proporsional mengikuti distribusi $P(Y)$.

### B. Traditional Machine Learning: LR, SVM, RF (TF-IDF)
* **Ekstraksi Fitur:** `TfidfVectorizer(max_features=10000)`.
* **Model Sesuai Implementasi Default Sklearn (Bab 3.6.1):**
  1. `LogisticRegression(max_iter=1000)`
  2. `SVC()` (Default SVM di Scikit-Learn menggunakan kernel RBF tanpa class weight, yang menjelaskan mengapa skor F1 SVM pada data teks imbalanced di paper sangat rendah, misal 0.042 di Issue).
  3. `RandomForestClassifier(n_estimators=100)`
* Setiap model dilatih secara single-task (terpisah per sumber data).

### C. Deep Learning: Single-Task Text-CNN (Kim 2014)
* **Embedding Layer:** `nn.Embedding(vocab_size, embed_dim=300)` diinisialisasi secara acak (scratch).
* **Convolutional Layers:** 3 filter ukuran window `[3, 4, 5]`, masing-masing 100 feature maps.
* **Activation & Pooling:** `ReLU` + 1D Global Max-Pooling.
* **Regularization & Optimizer:** Dropout `0.5`, Adam optimizer (`lr=1e-3`), CrossEntropyLoss.
* **Epochs:** `EPOCHS = 10` (Sesuai baseline Text-CNN Kim (2014) & Zhang & Wallace (2017) di mana Text-CNN konvergen pada 5–10 epoch).
* **Output:** Linear layer ke 5 kelas logits.
* Model dilatih terpisah 4 kali untuk masing-masing sumber data.

### D. Deep Learning: Multitask Text-CNN (MT-Text-CNN)
* **Shared Representation:** Layer embedding acak (`embed_dim=300`) dan shared convolutional filters digunakan bersama oleh ke-4 sumber data.
* **Task-Specific Heads:** Memiliki 4 linear classifier output terpisah (`fcs[0]` s/d `fcs[3]`).
* **Epochs:** `EPOCHS = 10`.
* **Stochastic Multitask Training (Collobert & Weston 2008):**
  1. Pada setiap langkah (*step*), pilih task secara acak $t \in \{\text{Comment}, \text{Commit}, \text{Pull}, \text{Issue}\}$.
  2. Ambil mini-batch dari task $t$.
  3. Lakukan forward pass melewati shared layer dan task head $t$.
  4. Hitung CrossEntropyLoss dan perbarui bobot jaringan melalui backpropagation.

---

## 6. Petunjuk Menjalankan Notebook (`reproduce_table4.ipynb`)

### A. Jika Dijalankan di Kaggle (Rekomendasi Terbaik: GPU T4 x2):
1. Buka [Kaggle](https://www.kaggle.com/) $\rightarrow$ klik **Create** $\rightarrow$ **New Notebook**.
2. Upload file `reproduce_table4.ipynb` (File $\rightarrow$ Import Notebook).
3. **PENTING (Setting di Panel Kanan / Sidebar Kaggle):**
   * **Accelerator:** Ubah dari *None* ke **GPU T4 x2** (atau *GPU T4*).
   * **Internet:** Wajib aktifkan switch ke **Internet On** (agar Cell 2 dapat mengunduh dataset langsung dari repositori GitHub).
4. Klik **Run All**.
   * Cell 2 akan otomatis meng-clone 4 file CSV dari repositori GitHub ke direktori kerja Kaggle (`/kaggle/working/data`).
   * Pelatihan 10-Fold CV Text-CNN & MT-Text-CNN akan berjalan cepat dengan akselerasi GPU Nvidia T4.

### B. Jika Dijalankan di Google Colab:
1. Buka [Google Colab](https://colab.research.google.com/) dan upload file `reproduce_table4.ipynb`.
2. Aktifkan GPU: **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ pilih **T4 GPU**.
3. Jalankan sel secara berurutan (*Run All*). Cell 2 otomatis mengunduh dataset dari GitHub.

### C. Jika Dijalankan di Laptop / Lokal:
1. Pastikan dependensi terpasang:
```bash
pip install torch torchvision scikit-learn pandas numpy tabulate nltk
```
2. Struktur folder lokal:
```
SATD-Research/
├── data/
│   ├── satd-dataset-code_comments.csv
│   ├── satd-dataset-commit_messages.csv
│   ├── satd-dataset-issues.csv
│   └── satd-dataset-pull_requests.csv
├── reproduce_table4.ipynb
└── STEP.md
```
3. Di **Cell 1**, terdapat konfigurasi:
   * `QUICK_TEST = True`: Menjalankan 2-fold cepat untuk verifikasi alur kerja pipeline tanpa menunggu lama.
   * `QUICK_TEST = False`: Menjalankan full 10-fold Cross-Validation persis sesuai eksperimen jurnal.
4. Jalankan seluruh sel secara berurutan (*Run All*).

---

## 7. Format Output Akhir (Sesuai Gambar Tabel 4 Paper)

Di akhir eksekusi Cell 10, notebook akan mencetak tabel persis seperti di jurnal:

```
Table 4: Comparison of F1-score between machine learning and baseline approaches
=====================================================================================
Classifier                    Type of SATD  Comment  Commit  Pull   Issue  Avg.   Average Imp. over Random
-- Deep Learning --
Text-CNN                      C/D.          0.665    0.485   0.515  0.461  0.531  7.8x
                              DOC.          0.526    0.632   0.484  0.456  0.524  52.4x
                              TST.          0.443    0.469   0.507  0.463  0.471  157.0x
                              REQ.          0.566    0.217   0.299  0.343  0.356  71.2x
                              AVG.          0.550    0.451   0.451  0.431  0.471  22.4x
MT-Text-CNN                   C/D.          0.725    0.536   0.539  0.486  0.571  8.4x
                              DOC.          0.626    0.659   0.441  0.457  0.546  54.6x
                              TST.          0.540    0.449   0.461  0.432  0.470  159.7x
                              REQ.          0.585    0.255   0.325  0.437  0.400  80.0x
                              AVG.          0.619    0.475   0.441  0.453  0.497  23.7x
-- Traditional Machine Learning --
LR                            C/D.          0.613    0.327   0.457  0.353  0.438  6.4x
...                           ...           ...      ...     ...    ...    ...    ...
SVM                           C/D.          0.400    0.051   0.085  0.008  0.136  2.0x
...                           ...           ...      ...     ...    ...    ...    ...
RF                            C/D.          0.600    0.199   0.095  0.065  0.240  3.5x
...                           ...           ...      ...     ...    ...    ...    ...
-- Baseline --
Random                        C/D.          0.053    0.071   0.071  0.076  0.068  -
                              DOC.          0.001    0.003   0.021  0.013  0.010  -
                              TST.          0.002    0.004   0.000  0.005  0.003  -
                              REQ.          0.009    0.000   0.004  0.005  0.005  -
                              AVG.          0.016    0.020   0.024  0.025  0.021  -
```
