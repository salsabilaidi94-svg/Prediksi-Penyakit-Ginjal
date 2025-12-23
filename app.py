import streamlit as st
import pandas as pd
import joblib
import base64

# =====================================================
# Page Config
# =====================================================
st.set_page_config(
    page_title="Prediksi CKD",
    page_icon="🩺",
    layout="centered"
)

# =====================================================
# Background Image
# =====================================================
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("c.jpg")

# =====================================================
# Advanced UI Styling
# =====================================================
st.markdown(
    """
    <style>
    /* Container utama */
    .block-container {
        background-color: rgba(255, 255, 255, 0.93);
        padding: 2.8rem;
        border-radius: 22px;
        max-width: 760px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.18);
    }

    /* Judul */
    h1 {
        font-weight: 800;
        color: #0d47a1;
        text-align: center;
    }

    h2, h3 {
        font-weight: 700;
        color: #1565c0;
        text-align: center;
    }

    /* Label form */
    label {
        font-weight: 700 !important;
        font-size: 15px !important;
        color: #0d47a1 !important;
    }

    /* Input angka */
    input[type="number"] {
        background-color: #e3f2fd !important;
        border-radius: 12px !important;
        padding: 10px !important;
        border: 1.5px solid #90caf9 !important;
        transition: all 0.2s ease-in-out;
    }

    input[type="number"]:focus {
        border: 2px solid #1976d2 !important;
        background-color: #ffffff !important;
        box-shadow: 0 0 0 3px rgba(25,118,210,0.25);
    }

    /* Selectbox */
    div[data-baseweb="select"] > div {
        background-color: #f7f9fc !important;
        border-radius: 12px !important;
        border: 1.5px solid #bdbdbd !important;
    }

    /* Tombol */
    button[kind="primary"] {
        background: linear-gradient(90deg, #1976d2, #42a5f5) !important;
        border-radius: 16px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        padding: 0.6rem 1.8rem !important;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }

    button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(25,118,210,0.45);
    }

    hr {
        margin: 1.8rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, #90caf9, transparent);
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# Load Model & Features
# =====================================================
model = joblib.load("model.pkl")      # pipeline (scaler + model)
features = joblib.load("features.pkl")

# =====================================================
# Header
# =====================================================
st.title("🩺 Prediksi Penyakit Ginjal Kronis")

# =====================================================
# Form Input
# =====================================================
with st.form("form_prediksi"):
    st.subheader("Data Pasien")

    col1, col2 = st.columns(2)

    with col1:
        bmi = st.number_input("**BMI**", min_value=0.0, step=0.1)
        alcohol = st.number_input("**Konsumsi Alkohol**", min_value=0.0, step=0.1)
        diet = st.number_input("**Kualitas Pola Makan**", min_value=0.0, step=0.1)
        fatigue = st.number_input("**Tingkat Kelelahan**", min_value=0.0, step=0.1)

    with col2:
        smoking = st.selectbox(
            "**Status Merokok**",
            [0, 1],
            format_func=lambda x: "Perokok" if x == 1 else "Tidak Perokok"
        )
        activity = st.number_input("Aktivitas Fisik", min_value=0.0, step=0.1)
        family_history = st.selectbox(
            "**Riwayat Keluarga Penyakit Ginjal**",
            [0, 1],
            format_func=lambda x: "Ada" if x == 1 else "Tidak Ada"
        )
        diabetes_medicine = st.selectbox(
            "**Mengonsumsi Obat Diabetes**",
            [0, 1],
            format_func=lambda x: "Ya" if x == 1 else "Tidak"
        )

    submit = st.form_submit_button("🔍 Prediksi")

# =====================================================
# Prediction
# =====================================================
if submit:
    input_df = pd.DataFrame([{
        "BMI": bmi,
        "Smoking": smoking,
        "AlcoholConsumption": alcohol,
        "PhysicalActivity": activity,
        "DietQuality": diet,
        "FamilyHistoryKidneyDisease": family_history,
        "AntidiabeticMedications": diabetes_medicine,
        "FatigueLevels": fatigue
    }])

    # Urutan fitur harus sama dengan training
    input_df = input_df[features]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.divider()
    st.subheader("🧪 Hasil Prediksi")

    # 0 = CKD | 1 = Tidak CKD
    if prediction == 0:
        st.markdown(
            f"""
            <div style="
                background-color:#ffebee;
                padding:22px;
                border-radius:16px;
                border-left:6px solid #d32f2f;
            ">
            <h3 style="color:#c62828;">⚠️ Terindikasi Penyakit Ginjal Kronis (CKD)</h3>
            <p><b>Probabilitas CKD:</b> {probability[0]*100:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="
                background-color:#e8f5e9;
                padding:22px;
                border-radius:16px;
                border-left:6px solid #2e7d32;
            ">
            <h3 style="color:#2e7d32;">✅ Tidak Terindikasi Penyakit Ginjal Kronis (CKD)</h3>
            <p><b>Probabilitas:</b> {probability[1]*100:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with st.expander("🔎 Lihat Data yang Diproses"):
        st.dataframe(input_df)

# =====================================================
# Footer
# =====================================================
st.markdown(
    "<hr><p style='text-align:center; font-size:12px;'>© 2025 | Aplikasi Prediksi CKD</p>",
    unsafe_allow_html=True
)
