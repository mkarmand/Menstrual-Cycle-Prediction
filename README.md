# 🩺 Prediksi Kategori Siklus Menstruasi

Aplikasi ini dibuat untuk memprediksi **kategori siklus menstruasi** berdasarkan berbagai data seperti panjang siklus, masa subur, skor menstruasi, serta faktor tambahan seperti stres, pola makan, kondisi medis, dan aktivitas fisik.

## 🎯 Tujuan

Memberikan informasi prediktif terkait kategori siklus:
- **Normal Pendek**: 21 - 25 hari  
- **Normal Sedang**: 26 - 30 hari  
- **Normal Panjang**: 31 - 35 hari  
- **Tidak Normal**: <21 atau >35 hari

## 📝 Fitur Input

- **Panjang Siklus**: Total hari dari awal menstruasi ke menstruasi berikutnya  
- **Hari Ovulasi & Luteal Phase**: Indikator fase subur dan tidak subur  
- **Masa Subur**: Hari-hari dengan kemungkinan ovulasi tinggi  
- **Jumlah Hari Masa Subur (Berdasarkan Rumus)**: Dihitung berdasarkan kombinasi lendir serviks, suhu tubuh, dan gejala lainnya  
- **Skor Menses**: Menggambarkan tingkat berat/ringan menstruasi tiap hari (1: ringan, 3: berat)  
- **Faktor Tambahan**:
  - **Stres**
  - **Pola Makan**
  - **Kondisi Medis**
  - **Aktivitas Fisik**  
  *Nilai skala 1–5 (1 = rendah, 5 = tinggi)*

## 📊 Output

- **Prediksi Model**: Kategori siklus berdasarkan model machine learning  
- **Klasifikasi Manual**: Berdasarkan panjang siklus yang dimasukkan  