# 🇮🇩 Dapur Data Indonesia
### Dashboard Harga Pangan Nasional · Powered by AI

---

## 📁 Struktur File

```
📂 project/
├── dashboard.py              ← File utama Streamlit
├── requirements.txt          ← Dependencies Python
├── dataset_dashboard.csv     ← Dataset harga pangan
├── model_rf_harga.pkl        ← Model Decision Tree Classifier
├── scaler_harga.pkl          ← MinMaxScaler (year, month)
├── label_encoders.pkl        ← LabelEncoder (admin1, market, category, commodity)
└── README.md
```

---

## ⚙️ Cara Menjalankan

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan dashboard
```bash
streamlit run dashboard.py
```

### 3. Buka di browser
```
http://localhost:8501
```

---

## 🤖 Fitur AI

Dashboard ini menggunakan **2 lapisan AI**:

### Layer 1 — Model Machine Learning (Decision Tree)
- Input: Provinsi, Komoditas, Tahun, Bulan
- Output: Label harga → **Murah / Sedang / Mahal**
- File: `model_rf_harga.pkl`
- Preprocessing: `scaler_harga.pkl` + `label_encoders.pkl`

### Layer 2 — Claude AI (Anthropic)
- Input: Hasil prediksi model + konteks historis
- Output: Analisis naratif dalam Bahasa Indonesia
- Diperlukan: **Anthropic API Key** (masukkan di sidebar)
- Endpoint: `claude-sonnet-4-20250514`

Cara mendapatkan API Key: https://console.anthropic.com

---

## 📊 Fitur Dashboard

| Tab | Konten |
|-----|--------|
| 📈 Tren Harga | Line chart & area chart perjalanan harga 2007-2023, distribusi label per tahun |
| 🗺️ Peta Provinsi | Bar chart ranking provinsi, heatmap komoditas x provinsi |
| 🤖 Prediksi AI | Prediksi label harga real-time + analisis Claude AI + batch 12 bulan |
| 🔥 Volatilitas & Insight | Ranking volatilitas, deteksi lonjakan harga, insight global Claude |

---

## 📋 Tentang Dataset

- **Sumber**: WFP / BPS Indonesia
- **Periode**: 2007 – 2023
- **Jumlah data**: 264.563 baris
- **Cakupan**: 35 provinsi, 30 komoditas, 7 kategori pangan
- **Label**: Murah / Sedang / Mahal (hasil klasifikasi model)
