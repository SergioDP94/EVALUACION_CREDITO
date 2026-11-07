import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de la página
st.set_page_config(page_title="Evaluador de Credit Scoring (Sin Pipeline)", page_icon="💳", layout="centered")

# Cargar los componentes independientes guardados previamente
@st.cache_resource
def cargar_componentes():
    modelo = joblib.load("modelo_logistico.joblib")
    imputer_num = joblib.load("imputer_num.joblib")
    scaler = joblib.load("scaler.joblib")
    imputer_cat = joblib.load("imputer_cat.joblib")
    encoder = joblib.load("encoder.joblib")
    return modelo, imputer_num, scaler, imputer_cat, encoder

modelo, imputer_num, scaler, imputer_cat, encoder = cargar_componentes()

cuantitativas = ["person_age", "person_income", "person_emp_length", "loan_amnt"]
cualitativas = ["person_home_ownership", "loan_intent"]

st.title("💳 Evaluador de Credit Scoring")
st.write("Ingrese los datos del solicitante. La transformación se realiza con componentes independientes (sin pipeline).")

# Formulario de entrada
with st.form("form_solicitante"):
    st.subheader("Datos del Solicitante")
    
    col1, col2 = st.columns(2)
    with col1:
        edad = st.number_input("Edad (años)", min_value=18, max_value=100, value=35)
        ingreso = st.number_input("Ingreso anual", min_value=0.0, value=65000.0, step=1000.0)
    with col2:
        antiguedad = st.number_input("Antigüedad laboral (años)", min_value=0.0, max_value=80.0, value=8.0)
        monto_prestamo = st.number_input("Monto del préstamo", min_value=0.0, value=8000.0, step=500.0)
    
    st.subheader("Detalles del Crédito y Vivienda")
    vivienda = st.selectbox("Tipo de vivienda", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    finalidad = st.selectbox("Finalidad del préstamo", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    
    boton_evaluar = st.form_submit_button("Evaluar Solicitud")

# Realizar predicción al enviar el formulario
if boton_evaluar:
    # 1. Crear el DataFrame con el nuevo cliente
    cliente = pd.DataFrame([{
        "person_age": edad,
        "person_income": ingreso,
        "person_emp_length": antiguedad,
        "loan_amnt": monto_prestamo,
        "person_home_ownership": vivienda,
        "loan_intent": finalidad
    }])
    
    # 2. Aplicar la transformación paso a paso (sin pipeline)
    num_procesado = scaler.transform(imputer_num.transform(cliente[cuantitativas]))
    cat_procesado = encoder.transform(imputer_cat.transform(cliente[cualitativas]))
    
    # Unir las matrices transformadas
    X_cliente = np.hstack([num_procesado, cat_procesado])
    
    # 3. Predecir con el modelo de clasificación directo
    probabilidad = modelo.predict_proba(X_cliente)[0, 1]
    clase = modelo.predict(X_cliente)[0]
    
    st.divider()
    st.subheader("Resultado de la Evaluación")
    
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.metric(label="Probabilidad de Incumplimiento", value=f"{probabilidad:.2%}")
    
    with col_res2:
        if clase == 1:
            st.error("⚠️ **Riesgo Alto:** Se predice incumplimiento.")
        else:
            st.success("✅ **Riesgo Bajo:** No se predice incumplimiento.")
