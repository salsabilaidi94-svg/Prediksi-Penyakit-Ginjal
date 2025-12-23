import streamlit as st
import pandas as pd
import joblib
import base64

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Prediksi CKD",
    page_icon="🩺",
    layout="centered"
)

# ===============================
# Background Image
# ===============================
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

        /* Card utama */
        .block-container {{
            background-color: rgba(255, 255, 255, 0.92);
            padding: 2.5rem;
            border-radius: 18px;
            max-width: 760px;
            box-shadow: 0 10px 35px rgba(0,0,0,0.18);
        }}

        h1, h2, h3 {{
            text-align: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("c.jpg")

# ===============================
# Custom CSS (FORM & UI)
# ===============================
st.markdown(
    """
    <style>
    /* Label jadi tebal */
    label {
        font-weight: 700 !important;
        font-size: 15px !important;
        color: #0d47a1 !important;
    }

    /* Input angka */
    input[type="number"] {
        background-color: #e3f2fd !important;
        border-radius: 10px !important;
        padding: 10px !important;
        border: 1px solid #64b5f6 !important;
        font-size: 14px !important;
    }

    /* Selectbox */
    div[data-baseweb="select"] > div {
        background-color: #f5f7fa !important;
        border-radius: 10px !important;
        border: 1px solid #b0bec5 !important;
        font-size: 14px !important;
    }

    /* Tombol prediksi */
    button[kind="primary"] {
        background: linear-gradient(135deg, #1976d2, #0d47a1) !important;
        border-radius: 12px !important;
        font-weight: bold !important;
        padding: 10px 18px !important;
        font-size: 16px !important;
    }

    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1565c0, #0b3c91) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# Load Model & Features
# ===============================
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

# ===============================
# Header
# ===============================
st.title("🩺 Prediksi Penyakit Ginjal Kronis")
# ===============================
# Form Input
# ===============================
with st.form("form_prediksi"):
    st.subheader("📋 Data Pasien")

    bmi = st.number_input("**BMI**", min_value=0.0, step=0.1)

    smoking = st.selectbox(
        "**Status Merokok**",
        [0, 1],
        format_func=lambda x: "Perokok" if x == 1 else "Tidak Perokok"
    )

    alcohol = st.number_input("**Konsumsi Alkohol**", min_value=0.0, step=0.1)
    activity = st.number_input("**Aktivitas Fisik**", min_value=0.0, step=0.1)
    diet = st.number_input("**Kualitas Pola Makan**", min_value=0.0, step=0.1)

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

    fatigue = st.number_input("**Tingkat Kelelahan**", min_value=0.0, step=0.1)

    submit = st.form_submit_button("🔍 Prediksi")

# ===============================
# Prediction
# ===============================
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

    input_df = input_df[features]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.divider()
    st.subheader("🧪 Hasil Prediksi")

    # ⚠️ Sesuaikan dengan label dataset
    # 0 = CKD | 1 = Tidak CKD
    if prediction == 0:
        st.error("⚠️ TERINDIKASI PENYAKIT GINJAL KRONIS (CKD)")
        st.progress(float(probability[0]))
        st.write(f"Probabilitas CKD: **{probability[0]*100:.2f}%**")
    else:
        st.success("✅ TIDAK TERINDIKASI PENYAKIT GINJAL KRONIS (CKD)")
        st.progress(float(probability[1]))
        st.write(f"Probabilitas Tidak CKD: **{probability[1]*100:.2f}%**")

    with st.expander("🔎 Detail Data Input"):
        st.dataframe(input_df)

# ===============================
# Footer
# ===============================
st.markdown(
    "<hr><p style='text-align:center; font-size:12px;'>© 2025 | Aplikasi Prediksi CKD</p>",
    unsafe_allow_html=True
)
