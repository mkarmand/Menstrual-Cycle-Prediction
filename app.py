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
- **Normal Pendek** (21–25 hari): Siklus lebih pendek, tapi masih dalam batas normal. Ovulasi cenderung terjadi lebih awal.
- **Normal Sedang** (26–30 hari): Siklus rata-rata perempuan. Ovulasi sekitar hari ke-14.
- **Normal Panjang** (31–35 hari): Siklus lebih panjang dari rata-rata, tetapi tetap normal. Ovulasi cenderung lebih lambat.
- **Tidak Normal** (<21 atau >35 hari): Bisa menandakan ketidakseimbangan hormon atau kondisi medis tertentu.
""")

# Formulir input
with st.form("form_prediksi"):
    st.subheader("Masukkan Informasi Siklus Anda:")

    cycle_peak = st.radio("Siklus Mengandung Puncak?", ["Ya", "Tidak"])
    length_cycle = st.number_input("Panjang Siklus (hari)", 18, 54)
    est_ovulation = st.number_input("Hari Ovulasi Perkiraan", 6, 29)
    luteal_phase = st.number_input("Panjang Fase Luteal", 1, 41)
    first_day_high = st.number_input("Hari Pertama Lendir Serviks masa Subur", 5, 26)
    total_high_days = st.number_input("Jumlah Hari Lendir masa Subur", 0, 22)
    total_high_post_peak = st.number_input("Lendir Serviks Subur Setelah Ovulasi", 0, 7)
    total_peak_days = st.number_input("Jumlah Hari Puncak pada Masa Subur", 0, 13)
    total_fertility_days = st.number_input("Total Hari Masa Subur", 0, 27)
    total_fertility_formula = st.number_input("Jumlah Hari Masa Subur (Rumus)", 6, 37)
    length_menses = st.number_input("Lama Menstruasi (hari)", 2, 15)
    num_intercourse = st.number_input("Jumlah Hari Berhubungan", 0, 20)
    intercourse_fertile = st.radio("Berhubungan saat Masa Subur?", ["Ya", "Tidak"])
    unusual_bleeding = st.radio("Ada Pendarahan Tidak Biasa?", ["Ya", "Tidak"])
    stress = st.slider("Skor Stres", 1, 5)
    diet = st.slider("Skor Pola Makan", 1, 5)
    medical = st.slider("Skor Kondisi Medis", 1, 5)

    submit = st.form_submit_button("Prediksi")

if submit:
    # Konversi radio ke angka
    cycle_peak = 1 if cycle_peak == "Ya" else 0
    intercourse_fertile = 1 if intercourse_fertile == "Ya" else 0
    unusual_bleeding = 1 if unusual_bleeding == "Ya" else 0

    # Susun input dalam DataFrame
    fitur_input = pd.DataFrame([[
        cycle_peak,
        length_cycle,
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
    ]], columns=[
        'CycleWithPeakorNot',
        'LengthofCycle',
        'EstimatedDayofOvulation',
        'LengthofLutealPhase',
        'FirstDayofHigh',
        'TotalNumberofHighDays',
        'TotalHighPostPeak',
        'TotalNumberofPeakDays',
        'TotalDaysofFertility',
        'TotalFertilityFormula',
        'LengthofMenses',
        'NumberofDaysofIntercourse',
        'IntercourseInFertileWindow',
        'UnusualBleeding',
        'StressScore',
        'DietScore',
        'MedicalConditionScore'
    ])

    # Prediksi tanpa scaling
    pred_proba = model.predict(fitur_input)
    # Jika output model berupa probabilitas, ambil argmax
    if pred_proba.ndim > 1 and pred_proba.shape[1] > 1:
        pred_label_idx = np.argmax(pred_proba, axis=1)[0]
    else:
        pred_label_idx = pred_proba[0]  # misal output langsung label

    label = encoder.inverse_transform([pred_label_idx])[0]

    # Tampilkan hasil
    st.success(f"Hasil Prediksi: **{label}**")

    # Deskripsi kategori
    deskripsi = {
        "Normal Pendek": "Siklus sedikit lebih pendek dari rata-rata. Umumnya tidak bermasalah kecuali disertai gejala lain.",
        "Normal Sedang": "Siklus normal dan stabil, umumnya menunjukkan keseimbangan hormonal yang baik.",
        "Normal Panjang": "Masih dalam batas normal. Siklus panjang bisa terjadi secara alami atau karena stres ringan.",
        "Tidak Normal": "Perlu pemantauan lebih lanjut. Siklus yang terlalu pendek atau panjang bisa menandakan masalah kesehatan."
    }

    st.info(f"**Keterangan:** {deskripsi.get(label, 'Kategori tidak dikenali')}")
