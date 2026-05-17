# 💨 WindPred — Prediksi Kecepatan Angin Jawa Barat

**Tugas UTS Praktikum Kecerdasan Buatan — Semester 4 (Genap) 2025/2026**

> Analisis komparatif 5 algoritma machine learning untuk prediksi kecepatan angin rata-rata harian (FF_AVG) berbasis data observasi BMKG Stasiun Klimatologi Jawa Barat.

---

## 👤 Identitas

| | |
|---|---|
| **Nama** | Aulia Nazwa Huriah |
| **NIM** | 301240010 |
| **Program Studi** | Teknik Informatika |
| **Mata Kuliah** | Praktikum Kecerdasan Buatan |

---

## 📊 Dataset & Sumber Data

Aplikasi ini menggunakan dua sumber data:

### 1. BMKG DataOnline *(data training)*
- **URL:** [dataonline.bmkg.go.id](https://dataonline.bmkg.go.id)
- **Stasiun:** Klimatologi Jawa Barat (ID WMO: 96753, 207 mdpl)
- **Rentang:** Mei 2024 – Mei 2026
- **Digunakan untuk:** Training & evaluasi semua model ML

### 2. Open-Meteo API *(data prediksi real-time)*
- **URL:** [open-meteo.com](https://open-meteo.com)
- **Endpoint:** `/v1/forecast` (hari ini & masa depan) dan `/v1/archive` berbasis ERA5 (historis)
- **Digunakan untuk:** Mengisi fitur cuaca otomatis saat prediksi by-date, tanpa perlu input manual
- **Lisensi:** Gratis untuk penggunaan non-komersial, tidak memerlukan API key

**Target Variable:** `FF_AVG` (kecepatan angin rata-rata harian, m/s)  
**Fitur Input:** TN, TX, TAVG, RH_AVG, RR, SS, FF_X, DDD_X, day_of_year, month, FF_AVG_lag1, FF_AVG_lag7, FF_AVG_roll7, time_index

---

## 🤖 Algoritma & Hasil

| Algoritma | Tipe | MAE ↓ | RMSE ↓ | R² ↑ | Evaluasi |
|---|---|---|---|---|---|
| Linear Regression | Supervised Regression | 0.6724 | 0.8685 | -0.1851 | Perlu Tuning |
| **ANN (TensorFlow)** ⭐ | Supervised Regression | **0.6065** | **0.7864** | **0.0284** | **Terbaik** |
| RNN/LSTM | Supervised Sequential | 0.6367 | 0.8318 | -0.0758 | Perlu Tuning |
| Backpropagation (NumPy) | Supervised Regression | 0.6307 | 0.7895 | 0.0207 | Normal |
| K-Means Clustering | Unsupervised | — | — | Silhouette: 0.3941 (K=3) | — |

> Model terbaik: **ANN** berdasarkan MAE, RMSE, dan R² tertinggi.  
> K-Means menghasilkan K=3 optimal (Angin Tenang, Angin Ringan, Angin Sedang) berdasarkan Silhouette Score — data Stasiun Klimatologi Jawa Barat tidak memiliki cluster angin kencang yang signifikan secara statistik.

---

## 🛠 Cara Instalasi & Menjalankan

```bash
# 1. Clone repository
git clone https://github.com/Alnazh/windpred.git
cd windpred

# 2. Install dependensi
pip install -r requirements.txt

# 3. Jalankan training (wajib sekali sebelum run app)
python train.py

# 4. Jalankan aplikasi
python -m flask --app app/app.py run

# Atau menggunakan gunicorn (production)
gunicorn app.app:app
```

Aplikasi berjalan di: `http://localhost:5000`

---

## 📁 Struktur Project

```
windpred/
├── data/
│   ├── bmkg_merged.csv          # Data mentah hasil merge
│   ├── bmkg_preprocessed.csv    # Data setelah feature engineering
│   └── clustered_data.csv       # Hasil K-Means clustering
├── models/
│   ├── linear_regression.pkl    # Model Linear Regression
│   ├── ann_model.keras/.h5      # Model ANN (TensorFlow)
│   ├── rnn_lstm_model.keras/.h5 # Model RNN/LSTM
│   ├── backprop_weights.npz     # Bobot Backpropagation manual
│   ├── kmeans_model.pkl         # Model K-Means
│   ├── scaler_X.pkl / scaler_y.pkl / scaler_kmeans.pkl
│   ├── model_metrics.json       # Metrik semua model
│   ├── kmeans_metrics.json      # Metrik & elbow K-Means
│   └── lstm_seq_buffer.json     # Buffer sekuens LSTM
├── notebooks/
│   └── windpred_eda_training.ipynb
├── app/
│   ├── static/                  # CSS, JS
│   ├── templates/               # HTML Jinja2
│   └── app.py                   # Flask application
├── train.py                     # Script training semua model
├── requirements.txt
├── Procfile
└── README.md
```

---

## 🔗 Links

| | |
|---|---|
| **Demo Aplikasi** | [ URL deploy ] |
| **Laporan PDF** | [ Link Google Classroom ] |
| **Video YouTube** | [ Link YouTube ] |

---

## 📄 Lisensi & Atribusi Data

| Sumber | Lisensi | Keterangan |
|---|---|---|
| **BMKG DataOnline** | Open Data Pemerintah | Bebas digunakan untuk keperluan akademik dan penelitian. [bmkg.go.id](https://www.bmkg.go.id) |
| **Open-Meteo** | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Data cuaca gratis berbasis ERA5 (Copernicus Climate Change Service). Tidak memerlukan API key untuk penggunaan non-komersial. [open-meteo.com](https://open-meteo.com) |
| **ERA5 (via Open-Meteo)** | Copernicus Climate Change Service | Reanalysis data dari ECMWF, diakses melalui Open-Meteo API. |
