
# Analisis Sentimen Ulasan Mobile Legends di Google Play Store

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange.svg)](https://www.tensorflow.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6-blueviolet.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Repositori ini berisi kode dan sumber daya untuk melakukan analisis sentimen pada ulasan game "Mobile Legends: Bang Bang" dari Google Play Store. Proyek ini mencakup seluruh alur kerja ilmu data, mulai dari pengumpulan data mentah, pemrosesan teks yang mendalam, hingga pelatihan dan perbandingan berbagai model machine learning dan deep learning.

![Word Cloud Ulasan](/foto.png)

## 📜 Tentang Proyek

Tujuan utama dari proyek ini adalah untuk menggali wawasan dari opini publik terhadap game Mobile Legends. Dengan menganalisis ratusan ribu ulasan, kita dapat memahami:

- **Persepsi Umum**: Apakah sentimen mayoritas pemain positif atau negatif?
- **Kritik & Pujian**: Aspek apa dari game yang paling sering dikeluhkan dan dipuji oleh pemain?
- **Performa Model**: Membandingkan efektivitas berbagai algoritma, dari machine learning klasik hingga model deep learning seperti LSTM, dalam memahami nuansa teks bahasa Indonesia.

Proyek ini dapat berguna bagi pengembang game untuk mendapatkan feedback, bagi pemain untuk melihat opini komunitas, atau sebagai portofolio untuk penerapan NLP (Natural Language Processing).

## 🛠️ Dibangun Dengan

Proyek ini mengandalkan beberapa pustaka dan framework utama:

- [**Google Play Scraper**](https://pypi.org/project/google-play-scraper/): Untuk mengambil data ulasan.
- [**Pandas**](https://pandas.pydata.org/): Untuk manipulasi dan analisis data.
- [**NLTK**](https://www.nltk.org/) & [**Sastrawi**](https://pypi.org/project/Sastrawi/): Untuk pemrosesan teks bahasa Indonesia (tokenisasi, stopword, stemming).
- [**Scikit-learn**](https://scikit-learn.org/): Untuk melatih model machine learning klasik.
- [**TensorFlow (Keras)**](https://www.tensorflow.org/): Untuk membangun dan melatih model deep learning (LSTM).
- [**Matplotlib**](https://matplotlib.org/) & [**WordCloud**](https://pypi.org/project/wordcloud/): Untuk visualisasi data.

## ⚙️ Struktur Proyek

Berikut adalah struktur direktori dari proyek ini:

```plaintext
├── 📄 ulasan_aplikasi.csv       # Dataset hasil scraping (output)
├── 📜 README.md                # File yang sedang Anda baca
├── 🐍 model.ipynb               # Notebook utama untuk preprocessing dan pemodelan
├── 🕷️ scrapping_dataset.ipynb  # Notebook untuk mengambil data ulasan
└── 📋 requirements.txt         # Daftar pustaka Python yang dibutuhkan
```

## Alur Kerja Proyek

Proyek ini dibagi menjadi beberapa tahap utama:

1.  **📥 Pengumpulan Data**:

    - Menggunakan `google-play-scraper`, kami mengambil **243.000 ulasan** dari aplikasi `com.mobile.legends`.
    - Fokus pada ulasan berbahasa Indonesia (`lang='id'`, `country='id'`).
    - Data mentah disimpan dalam file `ulasan_aplikasi.csv`.

2.  **🧹 Pemrosesan Teks (Preprocessing)**:
    Ini adalah tahap krusial untuk membersihkan dan menstandarisasi teks ulasan.

    - **Pembersihan Teks**: Menghapus URL, mention (`@`), tagar (`#`), angka, dan tanda baca.
    - **Case Folding**: Mengubah semua teks menjadi huruf kecil.
    - **Tokenisasi**: Memecah kalimat menjadi kata-kata (token).
    - **Normalisasi Kata Slang**: Mengganti kata-kata tidak baku dengan bentuk formalnya (misal: 'bgt' -> 'banget').
    - **Stopword Removal**: Menghapus kata-kata umum yang tidak memiliki makna sentimen (misal: 'yang', 'di', 'dan').

3.  **🏷️ Pelabelan Sentimen (Berbasis Leksikon)**:

    - Untuk data latih, sentimen (positif/negatif) pada awalnya ditentukan menggunakan kamus leksikon bahasa Indonesia.
    - Sebuah ulasan dilabeli 'positif' jika mengandung lebih banyak kata positif daripada negatif, dan sebaliknya. Ini menjadi dasar (ground truth) untuk model supervised learning.

4.  **🤖 Pelatihan Model**:
    Kami melatih beberapa model untuk membandingkan performanya:

    - **Machine Learning Klasik**:
      - Regresi Logistik
      - K-Nearest Neighbors (KNN)
      - Random Forest
    - **Deep Learning**:
      - Long Short-Term Memory (LSTM)

5.  **📊 Evaluasi**:
    Setiap model dievaluasi menggunakan metrik standar seperti Akurasi, Presisi, Recall, dan F1-Score untuk menentukan model mana yang paling efektif dalam mengklasifikasikan sentimen.

## 📈 Hasil

Berikut adalah perbandingan performa dari model-model yang diuji. Model dievaluasi pada data uji (test set) yang belum pernah dilihat sebelumnya.

| Model                    |  Akurasi  |  Presisi  |  Recall   | F1-Score  |
| ------------------------ | :-------: | :-------: | :-------: | :-------: |
| Regresi Logistik         |  _xx.x%_  |  _xx.x%_  |  _xx.x%_  |  _xx.x%_  |
| K-Nearest Neighbors      |  _xx.x%_  |  _xx.x%_  |  _xx.x%_  |  _xx.x%_  |
| Random Forest            |  _xx.x%_  |  _xx.x%_  |  _xx.x%_  |  _xx.x%_  |
| **LSTM (Deep Learning)** | **xx.x%** | **xx.x%** | **xx.x%** | **xx.x%** |

**\*Catatan:** Anda perlu mengisi nilai akurasi, presisi, recall, dan F1-score yang Anda dapatkan dari hasil eksekusi notebook `model.ipynb`.\*

## 🚀 Memulai

Untuk menjalankan proyek ini di mesin lokal Anda, ikuti langkah-langkah berikut.

### Prasyarat

Pastikan Anda memiliki Python (versi 3.9 atau lebih baru) dan `pip` terinstal di sistem Anda.

### Instalasi

1.  **Clone repositori ini:**

    ```sh
    git clone [https://github.com/nama-pengguna-anda/nama-repo-anda.git](https://github.com/nama-pengguna-anda/nama-repo-anda.git)
    cd nama-repo-anda
    ```

2.  **Buat dan aktifkan _virtual environment_ (sangat disarankan):**

    ```sh
    # Buat environment
    python -m venv venv

    # Aktifkan di Windows
    venv\Scripts\activate

    # Aktifkan di macOS/Linux
    source venv/bin/activate
    ```

3.  **Instal semua pustaka yang diperlukan:**
    ```sh
    pip install -r requirements.txt
    ```

## Penggunaan

1.  **Jalankan Scraper (Opsional)**:
    Jika Anda ingin mengambil data ulasan terbaru, buka dan jalankan semua sel di dalam `scrapping_dataset.ipynb`. Ini akan menimpa file `ulasan_aplikasi.csv`.

2.  **Jalankan Analisis & Pemodelan**:
    Buka `model.ipynb` dan jalankan sel-sel secara berurutan untuk melihat seluruh proses, mulai dari memuat data, preprocessing, hingga melatih dan mengevaluasi model. Di bagian akhir, Anda juga bisa mencoba memprediksi sentimen dari kalimat Anda sendiri.

## 🤝 Berkontribusi

Kontribusi adalah hal yang membuat komunitas sumber terbuka menjadi tempat yang luar biasa untuk belajar, menginspirasi, dan berkreasi. Setiap kontribusi yang Anda berikan sangat **dihargai**.

Jika Anda memiliki saran untuk membuat proyek ini lebih baik, silakan fork repo dan buat pull request. Anda juga bisa membuka issue dengan tag "enhancement".

1.  Fork Proyek
2.  Buat Branch Fitur Anda (`git checkout -b fitur/FiturLuarBiasa`)
3.  Commit Perubahan Anda (`git commit -m 'Menambahkan FiturLuarBiasa'`)
4.  Push ke Branch (`git push origin fitur/FiturLuarBiasa`)
5.  Buka Pull Request

## 📄 Lisensi

Didistribusikan di bawah Lisensi MIT. Lihat `LICENSE.txt` untuk informasi lebih lanjut.

## ✉️ Kontak

Nama Anda - [@handle_twitter_anda](https://twitter.com/handle_twitter_anda) - email@contoh.com

Link Proyek: [https://github.com/nama-pengguna-anda/nama-repo-anda](https://github.com/nama-pengguna-anda/nama-repo-anda)
=======
# sentimen_mobilelegends
>>>>>>> 1fcd8d84909a82609bc744b1b52591f5217d0870
