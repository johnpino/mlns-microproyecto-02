from pathlib import Path

import joblib
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "pipe.joblib"
pipe = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Clasificación ODS", page_icon="🌍")
st.title("Clasificación de texto por ODS")

ODS_NAMES = {
    1: "Fin de la pobreza",
    2: "Hambre cero",
    3: "Salud y bienestar",
    4: "Educación de calidad",
    5: "Igualdad de género",
    6: "Agua limpia y saneamiento",
    7: "Energía asequible y no contaminante",
    8: "Trabajo decente y crecimiento económico",
    9: "Industria, innovación e infraestructura",
    10: "Reducción de las desigualdades",
    11: "Ciudades y comunidades sostenibles",
    12: "Producción y consumo responsables",
    13: "Acción por el clima",
    14: "Vida submarina",
    15: "Vida de ecosistemas terrestres",
    16: "Paz, justicia e instituciones sólidas",
    17: "Alianzas para lograr los objetivos",
}

texto = st.text_area("Ingrese un texto", height=180)

if st.button("Predecir"):
    if texto.strip():
        pred = pipe.predict([texto])[0]
        st.success(f"ODS predicho: {pred} — {ODS_NAMES.get(pred, 'ODS')}")
    else:
        st.warning("Ingrese un texto antes de predecir.")
