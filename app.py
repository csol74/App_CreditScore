import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Predictor de Credit Score",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CARGA DEL MODELO Y SCALER
# ─────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    ruta = "modelo_ANN_multiclass_M2.keras"
    if not os.path.exists(ruta):
        return None
    return load_model(ruta)

@st.cache_resource
def cargar_scaler():
    ruta = "scaler_ANN_multiclass_M2.pkl"
    if not os.path.exists(ruta):
        return None
    return joblib.load(ruta)

model  = cargar_modelo()
scaler = cargar_scaler()

# ─────────────────────────────────────────────
# OPCIONES CATEGÓRICAS
# ─────────────────────────────────────────────
OPCIONES = {
    "Occupation": ["Scientist", "Teacher", "Engineer", "Entrepreneur", "Developer",
                   "Lawyer", "Media_Manager", "Doctor", "Journalist", "Manager",
                   "Accountant", "Musician", "Mechanic", "Writer", "Architect"],
    "Credit_Mix": ["Bad", "Standard", "Good"],
    "Payment_of_Min_Amount": ["Yes", "No"],
    "Payment_Behaviour": [
        "High_spent_Small_value_payments", "Low_spent_Large_value_payments",
        "High_spent_Medium_value_payments", "Low_spent_Small_value_payments",
        "High_spent_Large_value_payments", "Low_spent_Medium_value_payments"
    ],
    "Month": ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
}

def encode_categoricals(row_dict):
    encoded = row_dict.copy()
    for col, opciones in OPCIONES.items():
        le = LabelEncoder()
        le.fit(opciones)
        encoded[col] = le.transform([row_dict[col]])[0]
    return encoded

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("💳 Credit Score")
    st.markdown("---")
    st.markdown("""
    ### ℹ️ Sobre la app
    Clasifica clientes en:

    🔴 **Malo (0)** – Alto riesgo  
    🔵 **Estándar (1)** – Riesgo moderado  
    🟢 **Bueno (2)** – Bajo riesgo

    ---
    **Arquitectura ANN:**  
    `21 features`  
    `→ 256 → 128 → 64 → 32`  
    `→ 3 clases (Softmax)`

    ---
    **Técnicas aplicadas:**
    - EarlyStopping
    - Class Weight Balancing
    - StandardScaler
    """)
    st.caption("Taller ANN – Riesgo Crediticio")

# ─────────────────────────────────────────────
# TÍTULO PRINCIPAL
# ─────────────────────────────────────────────
st.title("💳 Predictor de Credit Score")
st.markdown("Ingresa los datos del cliente para predecir su categoría de crédito.")
st.markdown("---")

# Verificar archivos
if model is None:
    st.error("⚠️ No se encontró `modelo_ANN_multiclass_M2.keras`. Colócalo en la misma carpeta.")
    st.stop()

if scaler is None:
    st.error("⚠️ No se encontró `scaler_ANN_multiclass_M2.pkl`. Colócalo en la misma carpeta.")
    st.stop()

st.success("✅ Modelo y scaler cargados correctamente.")

# ─────────────────────────────────────────────
# FORMULARIO
# ─────────────────────────────────────────────
st.subheader("📋 Información del Cliente")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**👤 Datos Personales**")
    age             = st.number_input("Edad", min_value=18, max_value=100, value=35)
    occupation      = st.selectbox("Ocupación", OPCIONES["Occupation"])
    month           = st.selectbox("Mes", OPCIONES["Month"])
    annual_income   = st.number_input("Ingreso Anual (USD)", min_value=0.0, value=50000.0, step=1000.0)
    monthly_salary  = st.number_input("Salario Mensual Neto (USD)", min_value=0.0, value=4000.0, step=100.0)
    monthly_balance = st.number_input("Balance Mensual (USD)", min_value=-10000.0, value=500.0, step=100.0)
    amount_invested = st.number_input("Monto Invertido Mensual (USD)", min_value=0.0, value=200.0, step=50.0)

with col2:
    st.markdown("**🏦 Cuentas y Créditos**")
    num_bank_accounts  = st.number_input("Nº Cuentas Bancarias", min_value=0, max_value=20, value=2)
    num_credit_card    = st.number_input("Nº Tarjetas de Crédito", min_value=0, max_value=20, value=3)
    num_of_loan        = st.number_input("Nº de Préstamos", min_value=0, max_value=20, value=1)
    outstanding_debt   = st.number_input("Deuda Pendiente (USD)", min_value=0.0, value=5000.0, step=100.0)
    credit_utilization = st.number_input("Ratio Utilización Crédito (%)", min_value=0.0, max_value=100.0, value=30.0)
    total_emi          = st.number_input("Total EMI por Mes (USD)", min_value=0.0, value=300.0, step=50.0)
    credit_history_age = st.number_input("Antigüedad Historial Crédito (años)", min_value=0.0, value=5.0, step=0.5)

with col3:
    st.markdown("**⚠️ Historial de Pagos**")
    interest_rate       = st.number_input("Tasa de Interés Promedio (%)", min_value=0.0, max_value=50.0, value=15.0)
    delay_from_due      = st.number_input("Días de Retraso Promedio", min_value=0, max_value=180, value=10)
    num_delayed_payment = st.number_input("Nº Pagos Retrasados", min_value=0, max_value=50, value=5)
    credit_mix          = st.selectbox("Mezcla de Crédito", OPCIONES["Credit_Mix"])
    payment_min         = st.selectbox("¿Paga el Mínimo?", OPCIONES["Payment_of_Min_Amount"])
    payment_behaviour   = st.selectbox("Comportamiento de Pago", OPCIONES["Payment_Behaviour"])

st.markdown("---")

# ─────────────────────────────────────────────
# BOTÓN DE PREDICCIÓN
# ─────────────────────────────────────────────
if st.button("🔍 Predecir Credit Score", use_container_width=True, type="primary"):

    datos_raw = {
        "Month":                    month,
        "Age":                      age,
        "Occupation":               occupation,
        "Annual_Income":            annual_income,
        "Monthly_Inhand_Salary":    monthly_salary,
        "Num_Bank_Accounts":        num_bank_accounts,
        "Num_Credit_Card":          num_credit_card,
        "Interest_Rate":            interest_rate,
        "Num_of_Loan":              num_of_loan,
        "Delay_from_due_date":      delay_from_due,
        "Num_of_Delayed_Payment":   num_delayed_payment,
        "Credit_Mix":               credit_mix,
        "Outstanding_Debt":         outstanding_debt,
        "Credit_Utilization_Ratio": credit_utilization,
        "Credit_History_Age":       credit_history_age,
        "Payment_of_Min_Amount":    payment_min,
        "Total_EMI_per_month":      total_emi,
        "Amount_invested_monthly":  amount_invested,
        "Payment_Behaviour":        payment_behaviour,
        "Monthly_Balance":          monthly_balance,
    }

    # Codificar y escalar
    datos_enc = encode_categoricals(datos_raw)
    df_input  = pd.DataFrame([datos_enc])
    X_scaled  = scaler.transform(df_input)

    # Predicción
    probs      = model.predict(X_scaled, verbose=0)[0]
    pred_class = int(np.argmax(probs))

    # ── RESULTADO ──
    st.markdown("---")
    st.subheader("📊 Resultado de la Predicción")

    if pred_class == 0:
        st.error("🔴 Credit Score: MALO — Alto riesgo crediticio.")
        st.write("Se recomienda revisar el historial de pagos y reducir la deuda pendiente.")
    elif pred_class == 1:
        st.info("🔵 Credit Score: ESTÁNDAR — Riesgo moderado.")
        st.write("El cliente tiene un perfil aceptable pero con oportunidades de mejora.")
    else:
        st.success("🟢 Credit Score: BUENO — Bajo riesgo crediticio.")
        st.write("El cliente muestra un excelente historial financiero.")

    # ── PROBABILIDADES ──
    st.subheader("📈 Probabilidades por Clase")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric("🔴 Malo (0)", f"{probs[0]*100:.1f}%")
        st.progress(float(probs[0]))

    with col_b:
        st.metric("🔵 Estándar (1)", f"{probs[1]*100:.1f}%")
        st.progress(float(probs[1]))

    with col_c:
        st.metric("🟢 Bueno (2)", f"{probs[2]*100:.1f}%")
        st.progress(float(probs[2]))

    # ── CONFIANZA ──
    confianza = float(np.max(probs)) * 100
    st.markdown(f"**🎯 Confianza del modelo:** `{confianza:.1f}%`")
    if confianza < 60:
        st.warning("⚠️ Confianza baja. El modelo tiene incertidumbre en esta predicción.")
    elif confianza >= 85:
        st.success("✅ Alta confianza en la predicción.")

    # ── DATOS INGRESADOS ──
    with st.expander("🔎 Ver datos ingresados"):
        st.dataframe(df_input, use_container_width=True)