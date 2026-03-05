from pathlib import Path

import joblib
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "pipe.joblib"
pipe = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Clasificación ODS", page_icon="🌍")
st.title("Clasificación de texto por ODS")

texto = st.text_area("Ingrese un texto", height=180)

if st.button("Predecir"):
    if texto.strip():
        pred = pipe.predict([texto])[0]
        st.success(f"ODS predicho: {pred}")
    else:
        st.warning("Ingrese un texto antes de predecir.")
