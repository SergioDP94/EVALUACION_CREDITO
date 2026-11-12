import json
from pathlib import Path
import warnings

import joblib
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning


CARPETA = Path(__file__).resolve().parent
VIVIENDAS = {
    "MORTGAGE": "Hipotecada",
    "OTHER": "Otra",
    "OWN": "Propia",
    "RENT": "Alquilada",
}
DESTINOS = {
    "DEBTCONSOLIDATION": "Consolidación de deuda",
    "EDUCATION": "Educación",
    "HOMEIMPROVEMENT": "Mejoras en vivienda",
    "MEDICAL": "Gastos médicos",
    "PERSONAL": "Gastos personales",
    "VENTURE": "Emprendimiento",
}


@st.cache_resource
def cargar_modelo(ruta):
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        return joblib.load(ruta)


def predecir(modelo, solicitud):
    fila = pd.DataFrame([solicitud], columns=modelo.feature_names_in_)
    # Convierte la antigüedad vacía a NaN para que el pipeline pueda imputarla.
    fila["person_emp_length"] = fila["person_emp_length"].astype(float)
    indice_incumplimiento = modelo.classes_.tolist().index(1)
    return {
        "clase_predicha": int(modelo.predict(fila)[0]),
        "probabilidad_incumplimiento": float(
            modelo.predict_proba(fila)[0, indice_incumplimiento]
        ),
    }


st.set_page_config(page_title="Simulador de riesgo crediticio", layout="wide")
st.title("Simulador de riesgo crediticio")
st.write("Estima la probabilidad de incumplimiento de una solicitud de préstamo.")

try:
    modelo = cargar_modelo(str(CARPETA / "pipeline_credit_scoring.joblib"))
    ejemplos = json.loads((CARPETA / "clientes_ejemplo.json").read_text(encoding="utf-8"))
except FileNotFoundError as error:
    st.error(f"Falta el archivo {Path(error.filename).name}. Descomprime todo el ZIP de la app.")
    st.stop()
except InconsistentVersionWarning:
    st.error("La versión de scikit-learn difiere de la del modelo. Instala requirements.txt.")
    st.stop()

categorias = (
    modelo.named_steps["preparar"]
    .named_transformers_["categoricas"]
    .named_steps["codificar"]
    .categories_
)
viviendas, destinos = (valores.tolist() for valores in categorias)
entrada, salida = st.columns([3, 2], gap="large")

with entrada:
    st.subheader("Solicitud")
    perfil = st.selectbox(
        "Perfil de ejemplo",
        options=range(len(ejemplos)),
        format_func=lambda indice: f"Cliente {indice + 1}",
        key="perfil",
    )
    ejemplo = ejemplos[perfil]
    antiguedad_ejemplo = ejemplo["person_emp_length"]

    with st.form("solicitud"):
        datos_personales, datos_prestamo = st.columns(2)
        with datos_personales:
            edad = st.number_input(
                "Edad (años)", min_value=1, value=ejemplo["person_age"], key=f"edad_{perfil}"
            )
            ingreso = st.number_input(
                "Ingreso anual",
                min_value=0.0,
                value=float(ejemplo["person_income"]),
                step=1000.0,
                key=f"ingreso_{perfil}",
            )
            antiguedad = st.number_input(
                "Antigüedad laboral (años)",
                min_value=0.0,
                value=None if antiguedad_ejemplo is None else float(antiguedad_ejemplo),
                step=1.0,
                placeholder="Sin dato",
                help="Puedes dejar el campo vacío. El pipeline imputará el valor.",
                key=f"antiguedad_{perfil}",
            )
        with datos_prestamo:
            monto = st.number_input(
                "Monto del préstamo",
                min_value=0.0,
                value=float(ejemplo["loan_amnt"]),
                step=500.0,
                key=f"monto_{perfil}",
            )
            vivienda = st.selectbox(
                "Tipo de vivienda",
                options=viviendas,
                index=viviendas.index(ejemplo["person_home_ownership"]),
                format_func=VIVIENDAS.get,
                key=f"vivienda_{perfil}",
            )
            destino = st.selectbox(
                "Destino del préstamo",
                options=destinos,
                index=destinos.index(ejemplo["loan_intent"]),
                format_func=DESTINOS.get,
                key=f"destino_{perfil}",
            )

        st.caption("El ingreso y el monto usan las unidades monetarias del dataset.")
        enviado = st.form_submit_button("Calcular probabilidad", type="primary")

with salida, st.container(border=True):
    st.subheader("Resultado")
    if enviado:
        solicitud = {
            "person_age": edad,
            "person_income": ingreso,
            "person_emp_length": antiguedad,
            "loan_amnt": monto,
            "person_home_ownership": vivienda,
            "loan_intent": destino,
        }
        resultado = predecir(modelo, solicitud)
        probabilidad = resultado["probabilidad_incumplimiento"]
        clase = resultado["clase_predicha"]
        etiqueta = "1 · Incumplimiento" if clase == 1 else "0 · Sin incumplimiento"

        st.metric("Probabilidad estimada de incumplimiento", f"{probabilidad:.2%}")
        st.markdown(f"**Clase predicha** · {etiqueta}")
        if antiguedad is None:
            st.caption("El pipeline imputó la antigüedad laboral.")
    else:
        st.write("Selecciona un perfil o edita sus datos y calcula la probabilidad.")
