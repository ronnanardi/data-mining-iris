# Laporan Data Mining — Dataset Iris

## Identitas
| | |
|---|---|
| **Nama** | _isi nama_ |
| **NIM** | _isi NIM_ |
| **Mata Kuliah** | Data Mining |
| **Tanggal** | _isi tanggal_ |

---

## Tugas 1: Perbandingan KNN vs Random Forest

### 1.1 Tujuan
Membandingkan performa algoritma K-Nearest Neighbor (KNN) dan Random Forest (RF) pada dataset Iris menggunakan metrik accuracy, precision, recall, dan F1-Score.

### 1.2 Metodologi
- Dataset: Iris (150 sampel, 4 fitur, 3 kelas)
- Pembagian data: 80% latih, 20% uji
- Normalisasi: StandardScaler
- Evaluasi: 10-Fold Cross Validation

### 1.3 Hasil

| Metrik | KNN (k=_) | Random Forest |
|--------|-----------|---------------|
| Accuracy | | |
| Precision | | |
| Recall | | |
| F1-Score | | |
| CV Mean | | |

### 1.4 Analisis
> _Isi kesimpulan perbandingan berdasarkan hasil script_

---

## Tugas 2: Fuzzy C-Means Clustering

### 2.1 Tujuan
Mengimplementasikan metode Fuzzy C-Means untuk clustering data Iris dan membandingkan hasilnya dengan label asli.

### 2.2 Metodologi
- Jumlah cluster: c = 3
- Fuzziness parameter: m = 2
- Metrik: Adjusted Rand Index, Silhouette Score, Fuzzy Partition Coefficient

### 2.3 Hasil

| Metrik | Nilai |
|--------|-------|
| Adjusted Rand Index | |
| Silhouette Score | |
| Fuzzy Partition Coefficient | |
| Iterasi konvergen | |

### 2.4 Analisis
> _Isi interpretasi hasil clustering, kelas mana yang paling mudah/sulit dibedakan_

---

## Tugas 3: Perbandingan RapidMiner vs Orange (Decision Tree)

### 3.1 Tujuan
Membandingkan performa tools data mining RapidMiner dan Orange dalam klasifikasi dataset Iris menggunakan Decision Tree.

### 3.2 Konfigurasi

| Parameter | RapidMiner | Orange |
|-----------|------------|--------|
| Criterion | Gini | Gini |
| Max Depth | 10 | Tidak dibatasi |
| Min Samples Split | 4 | 2 |
| Min Samples Leaf | 2 | 1 |

### 3.3 Hasil

| Metrik | RapidMiner | Orange |
|--------|------------|--------|
| Accuracy | | |
| Precision | | |
| Recall | | |
| F1-Score | | |
| CV Mean | | |
| Tree Depth | | |
| Jumlah Daun | | |

### 3.4 Analisis
> _Isi perbandingan: kelebihan/kekurangan masing-masing tools_

---

## Kesimpulan Umum
> _Isi kesimpulan keseluruhan dari tiga tugas_

---

## Referensi
- Fisher, R.A. (1936). The use of multiple measurements in taxonomic problems.
- Scikit-learn documentation: https://scikit-learn.org
- Bezdek, J.C. (1981). Pattern Recognition with Fuzzy Objective Function Algorithms.
