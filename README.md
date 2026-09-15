# ECG Classification with 1D CNN

Notebook ini membangun model **1D Convolutional Neural Network (CNN)** untuk mengklasifikasikan sinyal ECG menjadi dua kelas:

- `Normal`
- `Abnormal`

Eksperimen mencakup eksplorasi data, preprocessing, pelatihan model, evaluasi, dan visualisasi confusion matrix.

## Struktur

```text
ecg/
|- ecg.ipynb
|- README.md
`- best_ecg_cnn.pth   # dibuat setelah proses training selesai
```

## Dataset

Notebook mengunduh dataset `ecg-dataset` dari Kaggle menggunakan `kagglehub`:

```text
devavratatripathy/ecg-dataset
```

File yang digunakan adalah `ecg.csv`. Kolom terakhir diperlakukan sebagai label, sedangkan kolom lainnya digunakan sebagai fitur sinyal ECG.

## Instalasi

Gunakan Python 3.9 atau yang lebih baru. Jalankan cell instalasi pada awal notebook:

```bash
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn torch
```

Pastikan environment memiliki akses internet dan konfigurasi Kaggle yang diperlukan agar `kagglehub` dapat mengunduh dataset.

## Menjalankan Notebook

1. Buka `ecg.ipynb` di Jupyter Notebook atau Visual Studio Code.
2. Jalankan semua cell secara berurutan.
3. Tunggu proses training selesai. Model terbaik disimpan sebagai `best_ecg_cnn.pth`.
4. Jalankan cell evaluasi untuk melihat accuracy, balanced accuracy, ROC-AUC, PR-AUC, MCC, classification report, dan confusion matrix.

## Pipeline

1. Mengunduh dan membaca dataset ECG.
2. Memeriksa bentuk data, missing values, dan distribusi label.
3. Membagi data secara stratified menjadi train, validation, dan test set.
4. Melakukan standardisasi menggunakan `StandardScaler` yang hanya di-fit pada data training.
5. Mengubah bentuk fitur menjadi format input CNN 1D: `(samples, channels, length)`.
6. Melatih CNN dengan AdamW, class-weighted cross-entropy, learning-rate scheduler, dan early stopping.
7. Memuat checkpoint dengan validation loss terbaik dan mengevaluasi performa pada test set.

## Arsitektur Model

Model `ECG_CNN` terdiri dari tiga convolutional block dengan batch normalization, ReLU, max pooling, dan dropout. Fitur tersebut diringkas dengan adaptive average pooling lalu diteruskan ke classifier berukuran `128 -> 64 -> 2`.

Seed eksperimen ditetapkan ke `42` untuk membantu reproduksibilitas.

## Catatan

Model ini dibuat untuk keperluan eksperimen dan pembelajaran. Hasil prediksi tidak boleh digunakan sebagai diagnosis medis tanpa validasi klinis dan pengawasan tenaga kesehatan.