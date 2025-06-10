import streamlit as st
import pandas as pd
import joblib

# Load model dan encoder
model = joblib.load('model.pkl')
encoder = joblib.load('encoder.pkl')

# Judul dan keterangan
st.title("Prediksi Kategori Siklus Menstruasi")
st.markdown("""
### Kategori Siklus:
- **Normal Pendek**: 21 - 25 hari  
- **Normal Sedang**: 26 - 30 hari  
- **Normal Panjang**: 31 - 35 hari  
- **Tidak Normal**: diluar 21 - 35 hari

### Keterangan:
- **Masa Subur**: Merujuk pada hari-hari yang berpotensi terjadi ovulasi.
- **Jumlah Hari Masa Subur (Berdasarkan Rumus)**: Total hari masa subur dihitung berdasarkan kombinasi lendir serviks, suhu tubuh, dan gejala lainnya.
- **Skor Menses**: Indikator ringan/sedang/beratnya menstruasi (1: ringan, 3: berat).
- **Faktor Tambahan (Stres, Pola Makan, Medis, Aktivitas Fisik)**:  
  Skala 1–5, dengan arti:  
  - **1** = Rendah/Ringan  
  - **5** = Tinggi/Berat  

""")

# Form input
with st.form("form_prediksi"):
    st.subheader("Masukkan Informasi Siklus:")

    # Panjang siklus
    length_of_cycle = st.number_input("Panjang Siklus (hari)", 18, 54)
    est_ovulation = st.number_input("Hari Ovulasi Perkiraan", 6, 29)
    luteal_phase = st.number_input("Panjang Fase Luteal", 1, 41)

    # Masa subur
    first_day_high = st.number_input("Hari Pertama Lendir Serviks Subur", 5, 26)
    total_high_days = st.number_input("Jumlah Hari Lendir Subur", 0, 22)
    total_high_post_peak = st.number_input("Lendir Subur Setelah Ovulasi", 0, 7)
    total_peak_days = st.number_input("Jumlah Hari Puncak Subur", 0, 13)
    total_fertility_days = st.number_input("Total Hari Masa Subur", 0, 27)
    total_fertility_formula = st.number_input("Jumlah Hari Masa Subur (Berdasarkan Rumus)", 6, 37)

    # Menses
    length_of_menses = st.number_input("Lama Menstruasi (hari)", 2, 15)
    menses_day1 = st.slider("Skor Hari Pertama Menstruasi", 1, 3)
    menses_day2 = st.slider("Skor Hari Kedua Menstruasi", 1, 3)
    menses_day3 = st.slider("Skor Hari Ketiga Menstruasi", 1, 3)
    menses_day4 = st.slider("Skor Hari Keempat Menstruasi", 1, 3)
    menses_day5 = st.slider("Skor Hari Kelima Menstruasi", 1, 3)
    total_menses_score = st.number_input("Total Skor Menstruasi", 2, 24)

    # Intercourse dan kondisi lain
    num_intercourse = st.number_input("Jumlah Hari Berhubungan", 0, 20)
    group = st.radio("Apakah Termasuk dalam Group?", ["Ya", "Tidak"])
    cycle_peak = st.radio("Apakah Siklus Mengandung Puncak?", ["Ya", "Tidak"])
    intercourse_fertile = st.radio("Berhubungan saat Masa Subur?", ["Ya", "Tidak"])
    unusual_bleeding = st.radio("Ada Pendarahan Tidak Biasa?", ["Ya", "Tidak"])

    # Faktor tambahan
    stress = st.slider("Skor Stres", 1, 5)
    diet = st.slider("Skor Pola Makan", 1, 5)
    medical = st.slider("Skor Kondisi Medis", 1, 5)
    activity = st.slider("Skor Aktivitas Fisik", 1, 5)

    submit = st.form_submit_button("Prediksi")

if submit:
    # Konversi radio ke angka
    group = 1 if group == "Ya" else 0
    cycle_peak = 1 if cycle_peak == "Ya" else 0
    intercourse_fertile = 1 if intercourse_fertile == "Ya" else 0
    unusual_bleeding = 1 if unusual_bleeding == "Ya" else 0

    # Susun input
    input_data = pd.DataFrame([{
        'Group': group,
        'CycleWithPeakorNot': cycle_peak,
        'LengthofCycle': length_of_cycle,
        'EstimatedDayofOvulation': est_ovulation,
        'LengthofLutealPhase': luteal_phase,
        'FirstDayofHigh': first_day_high,
        'TotalNumberofHighDays': total_high_days,
        'TotalHighPostPeak': total_high_post_peak,
        'TotalNumberofPeakDays': total_peak_days,
        'TotalDaysofFertility': total_fertility_days,
        'TotalFertilityFormula': total_fertility_formula,
        'LengthofMenses': length_of_menses,
        'MensesScoreDayOne': menses_day1,
        'MensesScoreDayTwo': menses_day2,
        'MensesScoreDayThree': menses_day3,
        'MensesScoreDayFour': menses_day4,
        'MensesScoreDayFive': menses_day5,
        'TotalMensesScore': total_menses_score,
        'NumberofDaysofIntercourse': num_intercourse,
        'IntercourseInFertileWindow': intercourse_fertile,
        'UnusualBleeding': unusual_bleeding,
        'StressScore': stress,
        'DietScore': diet,
        'MedicalConditionScore': medical,
        'PhysicalActivityScore': activity
    }])

    # Drop panjang siklus jika tidak digunakan oleh model
    input_data_for_model = input_data.drop(columns=['LengthofCycle'])

    # Prediksi
    prediction = model.predict(input_data_for_model)[0]
    label = encoder.inverse_transform([prediction])[0]

    # Hasil prediksi
    st.success(f"Model memprediksi siklus Anda masuk kategori: **{label}**")

    # Klasifikasi manual
    def classify_cycle(length):
        if 21 <= length <= 25:
            return 'Normal Pendek'
        elif 26 <= length <= 30:
            return 'Normal Sedang'
        elif 31 <= length <= 35:
            return 'Normal Panjang'
        else:
            return 'Tidak Normal'

    manual_kategori = classify_cycle(length_of_cycle)
    st.info(f"Berdasarkan panjang siklus yang Anda masukkan ({length_of_cycle} hari), secara manual diklasifikasikan sebagai: **{manual_kategori}**")
