# Simulador de riesgo crediticio con Streamlit

La app carga el pipeline exportado en la clase 1. Edita los datos de un solicitante
y obtén la probabilidad estimada de incumplimiento y la clase predicha.

## Ejecutar en Windows

Abre la terminal en esta carpeta y usa Python 3.12.

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
~~~

## Ejecutar en macOS o Linux

~~~bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m streamlit run app.py
~~~

Abre la URL local que aparece en la terminal.

## Probar el modelo

Selecciona cada perfil y pulsa **Calcular probabilidad**. Con los datos originales
de los ejemplos obtendrás estos resultados.

| Perfil | Probabilidad estimada | Clase |
|---|---:|---:|
| Cliente 1 | 6.32 % | 0 |
| Cliente 2, con antigüedad laboral vacía | 73.89 % | 1 |

Puedes modificar los seis campos y volver a calcular. El archivo
**predicciones_referencia.csv** contiene los valores completos.

## Cómo se conecta con el modelo

**app.py** carga **pipeline_credit_scoring.joblib** una vez y lo reutiliza.
La función **predecir()** convierte el formulario en una fila de pandas y llama
a **predict()** y **predict_proba()**. El pipeline contiene la imputación, el
escalado, la codificación de categorías y la regresión logística entrenada.

**requirements.txt** fija las versiones del modelo y de Streamlit.
**entorno.json** registra la versión de Python, las variables y el commit del CSV.
