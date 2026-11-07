import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración básica de la página
st.set_page_config(
    page_title="Evaluación de Riesgo Crediticio",
    page_icon="💳",
    layout="centered"
)

# Cargar el pipeline entrenado
@st.cache_resource
def cargar_pipeline():
    return joblib.load("pipeline_credit_scoring.joblib")

try:
    pipeline = cargar_pipeline()
except Exception as e:
    st.error(f"Error al cargar el archivo 'pipeline_credit_scoring.joblib': {e}")
    st.stop()

st.title("💳 Evaluación de Solicitud de Crédito")
st.write(
    "Ingresa los datos del solicitante para estimar la probabilidad de riesgo "
    "y conocer el estado de aprobación."
)

st.markdown("---")

# Formulario de entrada de datos
with st.form("formulario_credito"):
    st.subheader("Datos del Solicitante")
    
    col1, col2 = st.columns(2)
    
    with col1:
        edad = st.number_input(
            "Edad (años):",
            min_value=18,
            max_value=100,
            value=30,
            step=1
        )
        
        ingreso = st.number_input(
            "Ingreso anual ($):",
            min_value=1.0,
            max_value=2000000.0,
            value=45000.0,
            step=1000.0
        )
        
        tiene_antiguedad = st.checkbox("¿Especificar antigüedad laboral?", value=True)
        if tiene_antiguedad:
            antiguedad = st.number_input(
                "Antigüedad laboral (años):",
                min_value=0.0,
                max_value=float(edad - 14),
                value=4.0,
                step=0.5
            )
        else:
            antiguedad = np.nan  # El pipeline imputará automáticamente la mediana
            
    with col2:
        monto_prestamo = st.number_input(
            "Monto del préstamo solicitado ($):",
            min_value=100.0,
            max_value=100000.0,
            value=8000.0,
            step=500.0
        )
        
        vivienda = st.selectbox(
            "Tipo de vivienda:",
            options=["RENT", "MORTGAGE", "OWN", "OTHER"]
        )
        
        finalidad = st.selectbox(
            "Finalidad del préstamo:",
            options=[
                "EDUCATION",
                "MEDICAL",
                "VENTURE",
                "PERSONAL",
                "DEBTCONSOLIDATION",
                "HOMEIMPROVEMENT"
            ]
        )

    boton_evaluar = st.form_submit_button("Evaluar Solicitud")

# Evaluación al enviar el formulario
if boton_evaluar:
    # Construcción del DataFrame con las mismas columnas del pipeline
    solicitud = pd.DataFrame([{
        "person_age": edad,
        "person_income": ingreso,
        "person_emp_length": antiguedad,
        "loan_amnt": monto_prestamo,
        "person_home_ownership": vivienda,
        "loan_intent": finalidad
    }])
    
    # Estimación de la probabilidad de incumplimiento (clase 1)
    prob_incumplimiento = pipeline.predict_proba(solicitud)[0, 1]
    prob_porcentaje = prob_incumplimiento * 100

    st.markdown("---")
    st.subheader("Resultado de la Evaluación")
    
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Probabilidad de Incumplimiento", f"{prob_porcentaje:.2f}%")
    
    # Condición solicitada: menor al 20%
    if prob_porcentaje < 20.0:
        st.balloons()
        st.success("🎉 ¡Felicidades! Tu solicitud de crédito ha sido **Pre-Aprobada**.")
        st.info("El nivel de riesgo se encuentra en un rango óptimo (menor al 20%).")
    else:
        st.warning("⚠️ Solicitud pendiente de revisión manual.")
        st.write(
            "El nivel de riesgo supera el umbral del 20% para aprobación automática. "
            "Se requiere un análisis adicional por parte del equipo de riesgos."
        )