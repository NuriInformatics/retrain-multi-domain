# Ringkasan Komprehensif Paper: Identifikasi Otomatis SATD Multi-Sumber

> **Paper Rujukan:**  
> **Li, Y., Soliman, M., & Avgeriou, P. (2023).**  
> *Automatic identification of self-admitted technical debt from four different sources.*  
> **Empirical Software Engineering (EMSE)**, 28(3), 65. DOI: [10.1007/s10664-023-10297-9](https://doi.org/10.1007/s10664-023-10297-9).

---

## 1. Konteks & Motivasi Riset

*Self-Admitted Technical Debt* (SATD) merujuk pada utang teknis yang secara eksplisit diakui dan dicatat oleh pengembang perangkat lunak melalui teks saat mengambil kompromi teknis atau jalan pintas (*shortcuts*). Sebagian besar literatur sebelumnya mengkaji SATD secara terisolasi pada satu artefak saja, yakni komentar kode sumber (*source code comments*) melalui tag seperti `TODO` atau `FIXME`, serta sistem pelacak tiket (*issue trackers*).

Namun, dalam praktik rekayasa perangkat lunak modern, diskusi teknis dan pengakuan utang tersebar luas di artefak kolaborasi lain, khususnya pesan komit (*commit messages*) dan tinjauan kode (*pull requests*). Hingga riset ini diterbitkan, belum tersedia metode otomatis yang mampu mendeteksi SATD dari *commits* dan *pull requests*, maupun pendekatan terpadu yang menyatukan keempat sumber artefak tersebut. Untuk mengatasi tantangan tersebut, penulis mengusulkan pendekatan berbasis *Deep Multi-Task Learning*, yaitu **Multitask Text Convolutional Neural Network (MT-Text-CNN)**.

---

## 2. Pemisahan & Penjelasan Research Questions (RQ1 – RQ4)

Penelitian ini dirancang secara sistematis untuk menjawab empat pertanyaan penelitian (*Research Questions*):

```text
┌───────────────────────────────────────────────────────────────────────────────┐
│                      RESEARCH QUESTIONS (YIKUN LI ET AL.)                     │
├─────────┬───────────────────────────────┬─────────────────────────────────────┤
│  Nomor  │ Rumusan Pertanyaan Riset      │ Fokus & Temuan Utama                │
├─────────┼───────────────────────────────┼─────────────────────────────────────┤
│ **RQ1** │ Akurasi Identifikasi SATD     │ Menguji efektivitas MT-Text-CNN     │
│         │ Lintas Empat Sumber           │ vs Single-Task & Traditional ML.    │
│         │                               │ Hasil: MT-Text-CNN F1 0.497 s/d     │
│         │                               │ 0.611 (unggul signifikan).          │
├─────────┼───────────────────────────────┼─────────────────────────────────────┤
│ **RQ2** │ Kata Kunci Paling Informatif  │ Ekstraksi leksikon penanda SATD di  │
│         │ pada Masing-Masing Sumber     │ tiap sumber via CNN feature weight. │
│         │                               │ Hasil: Issue & PR memiliki irisan   │
│         │                               │ leksikon tertinggi; Comment paling  │
│         │                               │ spesifik.                           │
├─────────┼───────────────────────────────┼─────────────────────────────────────┤
│ **RQ3** │ Kuantitas & Tipe SATD         │ Mengukur prevalensi SATD pada 103   │
│         │ di Proyek Open Source Nyata   │ proyek Apache (30,3 juta teks).     │
│         │                               │ Hasil: Ditemukan 1,05 juta SATD;    │
│         │                               │ Code/Design Debt mendominasi (60%). │
├─────────┼───────────────────────────────┼─────────────────────────────────────┤
│ **RQ4** │ Relasi Keterhubungan SATD     │ Menelusuri korelasi dan siklus hidup│
│         │ Lintas Artefak Berbeda        │ SATD antar-sumber.                  │
│         │                               │ Hasil: Ditemukan 4 relasi:          │
│         │                               │ Introduction, Repayment, Tracking,  │
│         │                               │ dan Evolution.                      │
└─────────┴───────────────────────────────┴─────────────────────────────────────┘
```

### A. RQ1: How to accurately identify self-admitted technical debt from different sources?
* **Tujuan & Motivasi:** Menjawab apakah arsitektur *multi-task learning* dapat memecahkan masalah kelangkaan data latih berlabel pada sumber tertentu (*Commit* dan *Pull Request*) dengan mentransfer representasi fitur semantik dari sumber yang kaya data (*Comment* dan *Issue*).
* **Metode Evaluasi:** Model diuji menggunakan *Stratified 10-Fold Cross-Validation* di empat domain data. Penulis menguji dua fase: (1) fase arsitektur dasar dengan inisialisasi bobot acak (*randomized vectors*), dan (2) fase optimasi menggunakan *FastText word embedding*, *weighted loss*, dan variasi *feature maps*.
* **Temuan Utama:** MT-Text-CNN konsisten menempati peringkat tertinggi dengan Macro F1 sebesar **0.497** pada embedding acak (mengungguli Single Text-CNN 0.471 dan ML Tradisional 0.117–0.274), serta melonjak ke **0.611** pada model optimal ber-FastText. Keunggulan terbesar terjadi pada artefak dengan volume sampel kecil (*Commit* dan *Pull*).

### B. RQ2: What are the most informative keywords to identify self-admitted technical debt from different sources?
* **Tujuan & Motivasi:** Mengidentifikasi kata kunci yang menjadi sinyal kuat keberadaan SATD di tiap domain untuk memahami gaya bahasa teknis pengembang.
* **Metode Evaluasi:** Penulis menggunakan teknik *backtracking* pada bobot layer keluaran (*output layer*) dan fitur konvolusi Text-CNN terlatih untuk melacak kata-kata pemicu klasifikasi positif.
* **Temuan Utama:** Pola kata kunci bervariasi menurut konteks kerja:
  * *Comment:* Berorientasi tugas langsung (`todo`, `fixme`, `hack`, `workaround`, `temporary`, `remove`).
  * *Commit:* Berorientasi perbaikan cepat/patch (`workaround`, `hack`, `temporary`, `fix later`, `clean up`).
  * *Pull Request:* Berorientasi rencana tinjauan tim (`todo`, `future work`, `temporary solution`, `tech debt`, `refactoring`).
  * *Issue:* Berorientasi tiket formal sistemik (`debt`, `refactoring`, `cleanup`, `redesign`, `improve performance`).
  * Sumber *Issue* dan *Pull Request* memiliki derajat kesamaan leksikal tertinggi, disusul oleh *Commit*, sedangkan *Comment* memiliki perbendaharaan kata yang paling terisolasi.

### C. RQ3: How much and what types of self-admitted technical debt are documented in the four sources?
* **Tujuan & Motivasi:** Menilai skala sebaran dan proporsi tipe-tipe utang teknis yang riil ada di industri perangkat lunak terbuka berskala besar.
* **Metode Evaluasi:** Mengaplikasikan model MT-Text-CNN terbaik (F1 = 0.611) untuk mengklasifikasi 30,3 juta segmen teks dari 103 repositori open source Apache Software Foundation.
* **Temuan Utama:** 
  * Ditemukan sebanyak **1.050.412 item SATD**.
  * SATD tersebar merata di keempat sumber dan tidak hanya terkonsentrasi di komentar kode.
  * **Code/Design Debt (`C/D.`)** adalah kategori yang paling dominan di semua sumber (mencakup 60–70% total utang).
  * Proyek yang menerapkan alur kerja berbasis *Pull Request* mendokumentasikan SATD secara signifikan lebih banyak di *Issue* dan *Pull Request* dibandingkan proyek yang menggunakan *direct commit*.

### D. RQ4: What are the relations between self-admitted technical debt in different sources?
* **Tujuan & Motivasi:** Memahami bagaimana satu item utang teknis saling bertaut, berkembang, atau diselesaikan saat berpindah antar-artefak dalam siklus pengembangan.
* **Metode Evaluasi:** Membangun keterhubungan data melalui ID Issue dan hash commit, menghitung kemiripan semantik teks dengan *Cosine Similarity* (vektor *Bag-of-Words*), serta melakukan anotasi kualitatif manual dengan tingkat kesepakatan tinggi (*Cohen's Kappa = +0.81*).
* **Temuan Utama:** Teridentifikasi **4 jenis relasi resmi di paper** (Bab 4.4, halaman 29–31):
  1. **Documenting existing SATD repeatedly / in additional sources:** Utang yang sudah dicatat di satu sumber (misal Issue) didokumentasikan ulang di sumber lain (seperti Pull Request atau komentar kode `TODO` sebagai reminder).
  2. **Discussing the solution of SATD in other sources:** Utang dilaporkan di Issue, tetapi perdebatan solusi teknis detailnya dilakukan di Pull Request.
  3. **Documenting the repayment of SATD in other sources:** Pesan Commit atau komentar kode mendokumentasikan pelunasan utang yang sebelumnya tercatat di Issue.
  4. **Paying back documentation debt in code comments:** Kasus khusus di mana utang dokumentasi dilunasi langsung dengan menambahkan komentar atau Javadoc di kode sumber.

---

## 3. Alur Eksperimen Lengkap dari Jurnal (*Experimental Workflow*)

Alur kerja eksperimental paper Li et al. (2023) dirancang melalui tujuh tahapan terintegrasi:

```text
[Tahap 1: Seleksi Subjek & Pengumpulan Korpus Mentah]
  └─ 103 Proyek Apache (>500 commits, >500 PRs, >500 issues) -> 30,3 Juta Teks
                               │
                               ▼
[Tahap 2: Penautan Data Antar-Sumber (Data Linking)]
  └─ Hubungkan Commit Hash, Issue Key, dan PR References
                               │
                               ▼
[Tahap 3: Konstruksi Dataset Benchmark Berlabel]
  └─ 95.455 Sampel (Comment: 62k, Commit: 5k, PR: 5k, Issue: 23k) | Cohen's Kappa = 0.74
                               │
                               ▼
[Tahap 4: Pra-pemrosesan Teks (Data Cleansing)]
  └─ Regex stripping (//, /*), lowercasing, TweetTokenizer, filtering digit & noise
                               │
                               ▼
[Tahap 5: Arsitektur Model Deep Multi-Task & Baselines]
  ├─ MT-Text-CNN: Shared Word Embedding (300-d) + Shared Conv2D + 4 Task Heads
  ├─ Single-Task Text-CNN (dilatih terpisah per domain)
  ├─ Traditional ML: TF-IDF + Logistic Regression, SVM (RBF), Random Forest
  └─ Baseline: Random Classifier (berbasis apriori class probability)
                               │
                               ▼
[Tahap 6: Prosedur Training Seimbang & Evaluasi K-Fold]
  ├─ Stochastic Multi-Task Training (Collobert & Weston, 2008)
  ├─ Weighted Cross-Entropy Loss (penalti kelas minoritas REQ & TST)
  └─ Stratified 10-Fold CV -> Precision, Recall, Macro F1, T-Test, Cliff's Delta
                               │
                               ▼
[Tahap 7: Eksperimen Analisis Pasca-Model]
  ├─ Feature Backtracking -> Ekstraksi Kata Kunci Informatif (RQ2)
  ├─ Large-Scale Model Inference pada 103 Proyek Open Source (RQ3)
  └─ Cosine Similarity & Qualitative Tracing -> Pemetaan 4 Relasi SATD (RQ4)
```

### Rincian Setiap Tahapan:

#### 1. Seleksi Sistem Subjek (Bab 3.2)
Penulis memilih ekosistem **Apache Software Foundation** karena reputasi kualitas kodenya yang tinggi, tata kelola komunitas yang matang, serta keterbukaan dokumentasi komunikasinya. Dari 2.000+ repositori Apache di GitHub, ditetapkan kriteria seleksi:
* Repositori kode sumber, riwayat commit, pull request, dan issue harus terbuka untuk publik.
* Minimal memiliki **500 issue tickets, 500 pull requests, dan 500 commit messages**.  
Dari kriteria tersebut, diperoleh **103 proyek Java** (termasuk *Airflow, Spark, Flink, Kafka, Camel, HBase*).

#### 2. Penautan Data Lintas Sumber (*Data Linking*, Bab 3.3)
Untuk memungkinkan pelacakan keterhubungan SATD (persiapan RQ4), artefak dihubungkan secara otomatis melalui:
* *Link A:* Judul atau deskripsi Pull Request yang mencantumkan *Issue Key* (misal: `Fixes #12769`).
* *Link B:* Pesan Commit yang merujuk nomor Issue atau nomor PR.
* *Link C:* Hash commit ter-merge yang diperbarui secara otomatis di repositori GitHub PR.

#### 3. Konstruksi Dataset Teranotasi & Taksonomi Label (Bab 3.5)
Untuk melatih dan menguji model, disiapkan dataset berlabel yang dikurasi secara manual dan divalidasi dengan uji reliabilitas antar-penilai (*Inter-rater reliability* menggunakan *Cohen's Kappa* = +0.74, tergolong *substantial agreement*):

#### Tabel 1: Komposisi Dataset Benchmark Berlabel (Tabel 3 Paper)

| Sumber Artefak | C/D. (Code/Design) | DOC. (Documentation) | TST. (Test) | REQ. (Requirement) | Other (Non-SATD) | Total Sampel |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Code Comments** | 2.703 | 54 | 85 | 757 | 58.676 | **62.275** |
| **Commit Messages** | 522 | 98 | 58 | 27 | 4.295 | **5.000** |
| **Pull Requests** | 510 | 101 | 68 | 20 | 4.301 | **5.000** |
| **Issue Trackers** | 2.169 | 487 | 338 | 97 | 20.089 | **23.180** |
| **Total Sampel** | **5.904** | **740** | **549** | **901** | **87.361** | **95.455** |

*Definisi Kelas:*
* `C/D.` : Defisit struktur kode, duplikasi, atau cacat arsitektur.
* `DOC.` : Dokumentasi yang kurang lengkap, usang, atau tidak tepat.
* `TST.` : Cakupan tes tidak memadai atau unit test yang di-skip/ditunda.
* `REQ.` : Fungsionalitas yang belum tuntas diimplementasikan.
* `Other`: Kelas negatif non-SATD (mencakup *defect debt*, *build debt*, dll.).

#### 4. Pra-pemrosesan Teks (*Data Cleansing*, Bab 3.4)
Teks mentah dibersihkan menggunakan pipeline standar script `satd_detector.py`:
1. Menghapus tanda pembuka/penutup komentar (`//`, `/*`, `*/`).
2. Mengubah teks ke huruf kecil (*case folding*).
3. Tokenisasi kalimat (`nltk.sent_tokenize`) dan tokenisasi kata (`nltk.TweetTokenizer`).
4. Menghapus token yang mengandung digit/angka, tanda baca (kecuali tanda seru `!` dan tanya `?`), serta simbol non-semantik (`#`, `_`, `/`, `>`, `<`, `@`).

#### 5. Pemodelan & Arsitektur Jaringan (Bab 3.6.1 & 3.6.5)
* **Traditional ML Baselines:** Ekstraksi fitur teks menggunakan TF-IDF Sklearn dengan parameter default, kemudian diklasifikasikan menggunakan *Logistic Regression* (LR), *Support Vector Classifier* (SVM kernel RBF), dan *Random Forest* (RF, 100 trees).
* **Single-Task Text-CNN (Kim, 2014):** Model konvolusi teks standar yang dilatih secara independen pada masing-masing dari keempat domain data.
* **Proposed MT-Text-CNN:**
  * *Shared Embedding Layer:* Matriks representasi kata 300 dimensi.
  * *Shared Convolutional Layer:* Filter konvolusi ukuran window `[3, 4, 5]` dengan 100/200 feature maps, aktivasi ReLU, dan 1D Global Max-Pooling untuk menangkap n-gram penting dari seluruh domain.
  * *Task-Specific Heads:* Empat output layer linear terpisah, masing-masing memetakan representasi bersama ke 5 probabilitas kelas spesifik domain.
  * *Stochastic Joint Training (Collobert & Weston, 2008):* Pelatihan berulang di mana setiap langkah memilih task secara stokastik dan melakukan pembaruan bobot shared encoder secara simultan.

#### 6. Strategi Ketimpangan Data & Validasi (Bab 3.6.6 & 3.6.7)
* **Weighted Cross-Entropy Loss:** Memberikan penalti bobot loss berbanding terbalik dengan frekuensi kelas di data latih, memastikan kesalahan prediksi pada kelas langka (`REQ.` dan `TST.`) dihukum lebih berat.
* **Stratified 10-Fold Cross-Validation:** Memecah data menjadi 10 partisi dengan mempertahankan rasio kelas, di mana 9 partisi digunakan untuk training dan 1 partisi untuk testing secara bergantian.
* **Metrik Evaluasi:** Presisi, Recall, dan **Macro F1-Score (AVG)** dihitung per fold. Signifikansi keunggulan model diuji dengan **Student's T-Test** ($p < 0.05$) dan **Cliff's delta** (kategori ukuran efek *large* jika $d \ge 0.43$).

---

## 4. Rangkuman Hasil Evaluasi Tabel 4 (Benchmark Utama)

Berikut adalah rekapitulasi F1-Score pada fase evaluasi baseline arsitektur (Tabel 4 jurnal):

#### Tabel 2: Perbandingan F1-Score Pendekatan Machine Learning dan Baseline (Tabel 4 Paper)

| Kategori | Classifier | Type of SATD | Comment | Commit | Pull | Issue | Avg. | Average Imp. over Random |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Deep Learning** | **MT-Text-CNN** | C/D. | 0.725 | 0.536 | 0.539 | 0.486 | **0.571** | 8.4× |
| | | DOC. | 0.626 | 0.659 | 0.441 | 0.457 | **0.546** | 54.6× |
| | | TST. | 0.540 | 0.449 | 0.461 | 0.432 | 0.470 | 159.7× |
| | | REQ. | 0.585 | 0.255 | 0.325 | 0.437 | **0.400** | 80.0× |
| | | **AVG.** | **0.619** | **0.475** | **0.441** | **0.453** | **0.497** | **23.7×** |
| | Text-CNN | C/D. | 0.665 | 0.485 | 0.515 | 0.461 | 0.531 | 7.8× |
| | | DOC. | 0.526 | 0.632 | 0.484 | 0.456 | 0.524 | 52.4× |
| | | TST. | 0.443 | 0.469 | 0.507 | 0.463 | **0.471** | 157.0× |
| | | REQ. | 0.566 | 0.217 | 0.299 | 0.343 | 0.356 | 71.2× |
| | | **AVG.** | **0.550** | **0.451** | **0.451** | **0.431** | **0.471** | **22.4×** |
| **Traditional ML** | LR | AVG. | 0.400 | 0.253 | 0.236 | 0.208 | 0.274 | 13.0× |
| | SVM | AVG. | 0.171 | 0.173 | 0.081 | 0.042 | 0.117 | 5.6× |
| | RF | AVG. | 0.494 | 0.238 | 0.109 | 0.083 | 0.231 | 11.0× |
| **Baseline** | Random | AVG. | 0.016 | 0.020 | 0.024 | 0.025 | 0.021 | 1.0× |

Hasil uji statistik (Tabel 5 paper) membuktikan bahwa perbedaan performa MT-Text-CNN terhadap seluruh metode pembanding memiliki nilai $p < 0.001$ dan *Cliff's delta* bernilai *large* (0.44 s/d 1.00), memastikan keunggulan arsitektur multi-task valid secara saintifik.

---

## 5. Kesimpulan & Implikasi Riset

1. **Efektivitas Multitask Learning:** MT-Text-CNN terbukti sukses mengintegrasikan empat sumber data berbeda dan memitigasi isu kelangkaan sampel pada *Commit* dan *Pull Request* melalui mekanisme *shared representation*.
2. **Implikasi bagi Praktisi:** Pengawasan utang teknis tidak boleh hanya bergantung pada pencarian kata kunci `TODO` pada komentar kode. Banyak utang arsitektural penting yang hanya tercatat pada diskusi *Issue* dan *Pull Request*.
3. **Implikasi bagi Peneliti:** Pendekatan *multi-task text classification* merupakan paradigma yang sangat menjanjikan dalam rekayasa perangkat lunak untuk artefak-artefak perangkat lunak yang berkarakteristik *low-resource*.
