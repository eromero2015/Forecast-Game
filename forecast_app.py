import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing as ETS

# Configuración inicial
st.title("Análisis de Pronósticos con Técnicas Múltiples")

# Pesos para el promedio ponderado
st.sidebar.header("Configurar Pesos para el Promedio Ponderado")
peso_pm = st.sidebar.slider("Peso para Promedio Móvil", 0.0, 1.0, 0.33)
peso_ses = st.sidebar.slider("Peso para Suavización Exponencial Simple", 0.0, 1.0, 0.33)
peso_ets = st.sidebar.slider("Peso para ETS", 0.0, 1.0, 0.33)

# Verificación de suma de pesos
pesos_suman_uno = np.isclose(peso_pm + peso_ses + peso_ets, 1.0)
if not pesos_suman_uno:
    st.sidebar.error("La suma de los pesos debe ser 1.0")

# Inicializar datos
periodos = 10
historical_data = np.random.randint(100, 200, size=20)  # Datos históricos iniciales para 20 periodos
if 'real_demand' not in st.session_state:
    st.session_state.real_demand = [None] * periodos
if 'current_period' not in st.session_state:
    st.session_state.current_period = 0

# Valor fijo para la Suavización Exponencial Simple
alpha = 0.2

# Crear DataFrame para almacenar los pronósticos
df = pd.DataFrame(index=range(1, periodos + 1), columns=["Promedio Móvil", "Suavización Exponencial", "ETS", "Ponderado", "Demanda Real"])

# Inicializar el valor de SES con el primer valor histórico
ses_value = historical_data[0]

# Calcular pronósticos utilizando diferentes métodos
for i in range(periodos):
    # Promedio Móvil de 3 periodos
    df.loc[i + 1, "Promedio Móvil"] = np.mean(historical_data[i:i + 3])

    # Suavización Exponencial Simple
    if i >= 1:
        ses_value = alpha * historical_data[i + 9] + (1 - alpha) * ses_value
    df.loc[i + 1, "Suavización Exponencial"] = ses_value

    # ETS
    model = ETS(historical_data[:i + 10], trend='add', seasonal='add', seasonal_periods=3)
    fit = model.fit()
    df.loc[i + 1, "ETS"] = fit.forecast(steps=1)[0]

    # Promedio Ponderado
    df.loc[i + 1, "Ponderado"] = (peso_pm * df.loc[i + 1, "Promedio Móvil"] +
                                  peso_ses * df.loc[i + 1, "Suavización Exponencial"] +
                                  peso_ets * df.loc[i + 1, "ETS"])

# Actualizar la columna de Demanda Real en el DataFrame
df["Demanda Real"] = st.session_state.real_demand

# Mostrar la tabla de pronósticos
st.subheader("Tabla de Pronósticos")
st.dataframe(df)

# Botón para generar demanda real para el siguiente periodo (habilitado/deshabilitado según la suma de pesos)
if st.button("Generar Demanda Real para el Siguiente Periodo", disabled=not pesos_suman_uno):
    if st.session_state.current_period < periodos:
        nueva_demanda = np.random.randint(100, 200)
        st.session_state.real_demand[st.session_state.current_period] = nueva_demanda
        st.session_state.current_period += 1

        # Actualizar la columna de Demanda Real en el DataFrame
        df["Demanda Real"] = st.session_state.real_demand

        # Calcular errores y señal de rastreo
        df["Error"] = df["Demanda Real"] - df["Ponderado"]
        df["Señal de Rastreo"] = df["Error"].cumsum() / df["Error"].std()

        # Mostrar tabla actualizada
        st.dataframe(df)

        # Gráfica de Pronósticos y Demanda Real
        st.subheader("Pronósticos vs Demanda Real")
        plt.figure(figsize=(10, 6))
        plt.plot(df["Ponderado"], label="Pronóstico Ponderado", linestyle='--')
        plt.plot(df["Demanda Real"], label="Demanda Real", marker='o')
        plt.legend()
        st.pyplot(plt)

        # Gráfica interactiva de la señal de rastreo con límites
        st.subheader("Señal de Rastreo")
        fig, ax = plt.subplots()
        ax.plot(df["Señal de Rastreo"], label="Señal de Rastreo", color="blue")
        ax.axhline(y=4, color='red', linestyle='--')
        ax.axhline(y=-4, color='red', linestyle='--')
        ax.legend()
        st.pyplot(fig)

        # Gráfica interactiva del error de pronóstico
        st.subheader("Error de Pronóstico")
        st.line_chart(df["Error"])

    else:
        st.warning("Se ha alcanzado el límite de 10 periodos.")

# Botón para reiniciar y limpiar la demanda real
if st.button("Reiniciar Todo"):
    st.session_state.real_demand = [None] * periodos
    st.session_state.current_period = 0
    # Borrar la tabla y gráficos
    df["Demanda Real"] = st.session_state.real_demand
    df["Error"] = [None] * periodos
    df["Señal de Rastreo"] = [None] * periodos







