# retrain-multi-domain

Repositori untuk reproduksi eksperimen identifikasi *Self-Admitted Technical Debt* (SATD) multi-domain / lintas sumber (*source code comments, commit messages, pull requests, issues*) berdasarkan paper Li et al. (2023).

Fokus utama:
1. **Reproduce:** Menjalankan ulang pipeline pelatihan dan evaluasi model (Traditional ML, Single-Task Text-CNN, dan Multitask Text-CNN).
2. **Summary:** Mendapatkan ringkasan hasil evaluasi dan metrik performa (F1-score) per domain.

## Struktur Direktori

```text
.
├── data/              # Dataset 4 domain SATD (.csv)
├── notebook/          # Jupyter notebook eksperimen & reproduksi model
├── reference.txt      # Referensi jurnal dan paper acuan
├── summary.md         # Ringkasan hasil reproduksi (WIP)
├── STEP.md            # Dokumentasi panduan teknis reproduksi
└── README.md          # Dokumentasi ringkas proyek
```

## Referensi
- **Paper:** Li, Y., Soliman, M., & Avgeriou, P. (2023). *Automatic identification of self-admitted technical debt from four different sources*. Empirical Software Engineering, 28(3), 65.
- **DOI:** [10.1007/s10664-023-10297-9](https://doi.org/10.1007/s10664-023-10297-9)
