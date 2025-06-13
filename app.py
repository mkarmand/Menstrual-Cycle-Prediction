import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model dan encoder
model = joblib.load('model.pkl')
encoder = joblib.load('encoder.pkl')

# Judul aplikasi
st.title("Prediksi Kategori Siklus Menstruasi")
st.markdown("""
### Kategori Siklus:
- **Normal Pendek** (21–25 hari)
- **Normal Sedang** (26–30 hari)
- **Normal Panjang** (31–35 hari)
- **Tidak Normal** (Diluar rentang 21-35 hari)

---

### Rumus Total Fertility (Jumlah Hari Lendir Subur):
Panjang Fase Subur (Berdasarkan Rumus) = Panjang Fase Lendir Serviks Subur (hari) + Panjang Fase Lendir Serviks Subur Setelah Ovulasi (hari) + Panjang Fase Puncak Subur (hari)

---

### Skor Penilaian:
- **Stres:**  
  1 = Rendah, 2 = Sedang Rendah, 3 = Sedang, 4 = Sedang Tinggi, 5 = Tinggi

- **Pola Makan:**  
  1 = Buruk, 2 = Kurang Baik, 3 = Cukup, 4 = Baik, 5 = Sangat Baik

- **Kondisi Medis:**  
  1 = Ringan/Tidak Ada, 2 = Sedikit, 3 = Sedang, 4 = Cukup Serius, 5 = Serius
""")


with st.form("form_prediksi"):
    st.subheader("Masukkan Informasi Siklus Anda:")

    cycle_peak = st.radio("Apakah Anda mendeteksi hari puncak kesuburan pada siklus ini?", ["Ya", "Tidak"])
    est_ovulation = st.number_input("Perkiraan Hari Ovulasi (hari)", 8, 29)
    luteal_phase = st.number_input("Panjang Fase Luteal (hari)", 1, 41)
    first_day_high = st.number_input("Hari ke-berapa pertama kali muncul lendir serviks subur?", 5, 25)
    total_high_days = st.number_input("Panjang Fase Lendir Serviks Subur (hari)", 0, 19)
    total_high_post_peak = st.number_input("Panjang Fase Lendir Serviks Subur Setelah Ovulasi (hari)", 0, 7)
    total_peak_days = st.number_input("Panjang Fase Puncak Subur (hari)", 0, 13)
    total_fertility_days = st.number_input("Panjang Fase Subur (hari)", 2, 27)
    total_fertility_formula = st.number_input("Panjang Fase Subur (Berdasarkan Rumus)", 6, 39)
    length_menses = st.number_input("Lama Menstruasi (hari)", 2, 15)
    num_intercourse = st.number_input("Jumlah Hari Berhubungan", 0, 19)
    intercourse_fertile = st.radio("Berhubungan saat Masa Subur?", ["Ya", "Tidak"])
    unusual_bleeding = st.radio("Pendarahan Tidak Biasa?", ["Ya", "Tidak"])
    stress = st.slider("Skor Stres", 1, 5)
    diet = st.slider("Skor Pola Makan", 1, 5)
    medical = st.slider("Skor Kondisi Medis", 1, 5)

    submit = st.form_submit_button("Prediksi")

if submit:
    # Ubah input radio ke angka
    cycle_peak = 1 if cycle_peak == "Ya" else 0
    intercourse_fertile = 1 if intercourse_fertile == "Ya" else 0
    unusual_bleeding = 1 if unusual_bleeding == "Ya" else 0

    # Buat dataframe input (tanpa LengthofCycle)
    fitur_input = pd.DataFrame([[
        cycle_peak,
        est_ovulation,
        luteal_phase,
        first_day_high,
        total_high_days,
        total_high_post_peak,
        total_peak_days,
        total_fertility_days,
        total_fertility_formula,
        length_menses,
        num_intercourse,
        intercourse_fertile,
        unusual_bleeding,
        stress,
        diet,
        medical
    ]])

    # Prediksi
    pred = model.predict(fitur_input)
    label = encoder.inverse_transform(pred)[0]

    # Tampilkan hasil
    st.success(f"Hasil Prediksi: **{label}**")

    # Penjelasan hasil
    deskripsi = {
        "Normal Pendek": "Siklus sedikit lebih pendek dari rata-rata. Umumnya tidak bermasalah kecuali disertai gejala lain.",
        "Normal Sedang": "Siklus normal dan stabil, umumnya menunjukkan keseimbangan hormonal yang baik.",
        "Normal Panjang": "Masih dalam batas normal. Siklus panjang bisa terjadi secara alami atau karena stres ringan.",
        "Tidak Normal": "Perlu pemantauan lebih lanjut. Siklus yang terlalu pendek atau panjang bisa menandakan masalah kesehatan."
    }

    st.info(f"**Keterangan:** {deskripsi.get(label, 'Kategori tidak dikenali')}")
