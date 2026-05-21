
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set page title
st.set_page_config(page_title='Prediksi Gaji Pertama Vokasi', layout='centered')

# 1. Load the pre-trained artifacts
@st.cache_resource
def load_artifacts():
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('standard_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('linear_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return encoders, scaler, model

try:
    encoders, scaler, model = load_artifacts()

    st.title("🚀 Aplikasi Prediksi Gaji Pertama")
    st.markdown("Masukkan data peserta pelatihan di bawah ini untuk mengestimasi gaji pertama (dalam Juta).")

    # 2. User Inputs
    col1, col2 = st.columns(2)

    with col1:
        jenis_kelamin = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Wanita'])
        usia = st.number_input('Usia', min_value=17, max_value=50, value=25)
        pendidikan = st.selectbox('Pendidikan', encoders['Pendidikan'].classes_)
        status_bekerja = st.selectbox('Status Bekerja', encoders['Status_Bekerja'].classes_)

    with col2:
        jurusan = st.selectbox('Jurusan', [j.capitalize() for j in encoders['Jurusan'].classes_])
        durasi_jam = st.number_input('Durasi Jam Pelatihan', min_value=1, max_value=100, value=60)
        nilai_ujian = st.number_input('Nilai Ujian', min_value=0, max_value=100, value=85)

    # 3. Preprocessing logic
    if st.button('Prediksi Gaji'):
        # Prepare input as DataFrame
        input_data = pd.DataFrame({
            'Jenis_Kelamin': [jenis_kelamin],
            'Usia': [usia],
            'Pendidikan': [pendidikan],
            'Jurusan': [jurusan.lower()],
            'Durasi_Jam': [durasi_jam],
            'Nilai_Ujian': [nilai_ujian],
            'Status_Bekerja': [status_bekerja]
        })

        # Apply manual mapping for Gender consistency
        input_data['Jenis_Kelamin'] = input_data['Jenis_Kelamin'].replace({'Laki-laki': 'L', 'Wanita': 'P'})

        # Apply Label Encoding
        for col in encoders:
            input_data[col] = encoders[col].transform(input_data[col])

        # Apply Scaling
        input_scaled = scaler.transform(input_data)

        # Make Prediction
        prediction = model.predict(input_scaled)

        # Show Result
        st.success(f"### Estimasi Gaji Pertama: Rp {prediction[0]:.2f} Juta")

except FileNotFoundError:
    st.error("File pickle tidak ditemukan. Pastikan 'label_encoders.pkl', 'standard_scaler.pkl', dan 'linear_regression_model.pkl' ada di direktori yang sama.")
