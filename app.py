import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ======================
# CONFIGURACIÓN GENERAL
# ======================
st.set_page_config(
    page_title="Cuadro de mando de accidentes 2019–2023",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# CARGA DE DATOS LIMPIOS
# ======================
@st.cache_data
def cargar_datos():
    vehiculos = pd.read_csv("./working_dir/Vehiculos_2019_2023_diccionario.csv", encoding="utf-8")
    hechos = pd.read_csv("./working_dir/Hechos_2019_2023_diccionario.csv", encoding="utf-8")
    fallecidos = pd.read_csv("./working_dir/Fallecidos_2019_2023_diccionario.csv", encoding="utf-8")
    return vehiculos, hechos, fallecidos

vehiculos, hechos, fallecidos = cargar_datos()

# ======================
# SIDEBAR PRINCIPAL
# ======================
st.sidebar.title("📊 Cuadro de mando interactivo")
modo = st.sidebar.radio(
    "Selecciona una sección:",
    ("Exploración de Datos", "Modelos de Predicción")
)

# ======================
# SECCIÓN 1: EXPLORACIÓN
# ======================
if modo == "Exploración de Datos":
    st.header("🔍 Exploración interactiva de datos")

    dataset = st.selectbox("Selecciona un conjunto de datos:", ["Vehículos", "Hechos", "Fallecidos"])
    df = {"Vehículos": vehiculos, "Hechos": hechos, "Fallecidos": fallecidos}[dataset]

    st.markdown("### 🎚️ Filtros de visualización")

    if "ano_ocu" in df.columns:
        anos = sorted(df["ano_ocu"].dropna().unique())
        ano_sel = st.selectbox("Selecciona año:", [None] + list(anos))
        if ano_sel:
            df = df[df["ano_ocu"] == ano_sel]

    if "sexo_per" in df.columns:
        sexos = sorted(df["sexo_per"].dropna().unique())
        sexo_sel = st.multiselect("Selecciona sexo:", sexos)
        if sexo_sel:
            df = df[df["sexo_per"].isin(sexo_sel)]

    if "depto_ocu" in df.columns:
        deptos = sorted(df["depto_ocu"].dropna().unique())
        depto_sel = st.multiselect("Selecciona departamento:", deptos)
        if depto_sel:
            df = df[df["depto_ocu"].isin(depto_sel)]

    if "mupio_ocu" in df.columns:
        mupios = sorted(df["mupio_ocu"].dropna().unique())
        mupio_sel = st.multiselect("Selecciona municipio:", mupios)
        if mupio_sel:
            df = df[df["mupio_ocu"].isin(mupio_sel)]

    st.markdown("### 📋 Primeras filas del dataset filtrado")
    st.dataframe(df.head(20))

# ======================
# SECCIÓN 2: MODELOS
# ======================
elif modo == "Modelos de Predicción":
    st.header("🤖 Modelos de predicción y clasificación")

    modelo_sel = st.selectbox(
        "Selecciona un modelo:",
        [
            "1️⃣ Predicción del sexo de la persona",
            "2️⃣ Predicción del grupo horario del accidente"
        ]
    )

    # --------------------------
    # MODELO 1: SEXO DE PERSONA
    # --------------------------
    if "sexo" in modelo_sel.lower():
        st.subheader("🧍‍♂️ Modelo 1: Predicción del sexo de la persona")

        df = fallecidos.copy().dropna(subset=["sexo_per"])
        features = ["depto_ocu", "zona_ocu", "tipo_eve", "tipo_veh", "g_hora_5"]
        df = df.dropna(subset=features)

        X = df[features].apply(LabelEncoder().fit_transform)
        y = LabelEncoder().fit_transform(df["sexo_per"])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))
        st.info(f"📈 Precisión general del modelo: {acc*100:.2f}%")

        st.markdown("### 🔢 Prueba una predicción manual")
        depto = st.selectbox("Departamento:", sorted(df["depto_ocu"].unique()))
        zona = st.selectbox("Zona:", sorted(df["zona_ocu"].unique()))
        tipo_eve = st.selectbox("Tipo de evento:", sorted(df["tipo_eve"].unique()))
        tipo_veh = st.selectbox("Tipo de vehículo:", sorted(df["tipo_veh"].unique()))
        g_hora = st.selectbox("Grupo horario:", sorted(df["g_hora_5"].unique()))

        if st.button("Predecir sexo"):
            entrada = pd.DataFrame([[depto, zona, tipo_eve, tipo_veh, g_hora]], columns=features)

            # Codificar igual que el entrenamiento
            entrada_enc = entrada.copy()
            for col in features:
                le = LabelEncoder()
                le.fit(df[col])
                entrada_enc[col] = le.transform(entrada[col])

            pred = model.predict(entrada_enc)[0]
            label_map = dict(enumerate(sorted(df["sexo_per"].unique())))
            st.success(f"Predicción: **{label_map[pred]}**")


    # --------------------------
    # MODELO 2: GRUPO HORARIO
    # --------------------------
    elif "horario" in modelo_sel.lower():
        st.subheader("⏰ Modelo 2: Predicción del grupo horario (mañana, tarde, noche)")

        df = hechos.copy().dropna(subset=["g_hora_5"])
        features = ["tipo_eve", "mes_ocu", "zona_ocu", "depto_ocu", "tipo_veh", "color_veh"]
        df = df.dropna(subset=features)

        X = df[features].apply(LabelEncoder().fit_transform)
        y = LabelEncoder().fit_transform(df["g_hora_5"])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=1)
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))
        st.info(f"📈 Precisión general del modelo: {acc*100:.2f}%")

        st.markdown("### 🔢 Prueba una predicción manual")
        tipo_eve = st.selectbox("Tipo de evento:", sorted(df["tipo_eve"].unique()))
        mes = st.selectbox("Mes de ocurrencia:", sorted(df["mes_ocu"].unique()))
        zona = st.selectbox("Zona:", sorted(df["zona_ocu"].unique()))
        depto = st.selectbox("Departamento:", sorted(df["depto_ocu"].unique()))
        tipo_veh = st.selectbox("Tipo de vehículo:", sorted(df["tipo_veh"].unique()))
        color = st.selectbox("Color del vehículo:", sorted(df["color_veh"].unique()))

        if st.button("Predecir grupo horario"):
            entrada = pd.DataFrame([[tipo_eve, mes, zona, depto, tipo_veh, color]], columns=features)

            # Codificar igual que el entrenamiento
            entrada_enc = entrada.copy()
            for col in features:
                le = LabelEncoder()
                le.fit(df[col])
                entrada_enc[col] = le.transform(entrada[col])

            pred = model.predict(entrada_enc)[0]
            label_map = dict(enumerate(sorted(df["g_hora_5"].unique())))
            st.success(f"Predicción: **{label_map[pred]}**")

