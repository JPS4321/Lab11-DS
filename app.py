import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


st.set_page_config(
    page_title="Informacion de accidentes 2019–2023",
    layout="wide",
    initial_sidebar_state="expanded"
)

PALETTE = ["#002B5B", "#0059B3", "#007ACC", "#FFA500", "#FF6600"]

st.markdown("""
    <style>
        body {
            background-color: #0E1117 !important; /* fondo oscuro principal */
            color: #FAFAFA !important;            /* texto claro */
        }
        .stApp {
            background-color: #0E1117;            /* mismo tono de fondo */
        }
        h1, h2, h3, h4, h5 {
            color: #00B4D8 !important;            /* azul brillante para títulos */
            font-weight: 650;
        }
        .stSidebar, .st-c2, .st-emotion-cache-1y4p8pa { 
            background-color: #111418 !important; /* sidebar más oscuro */
        }
        .metric-label, .metric-value {
            color: #FAFAFA !important;            /* texto de métricas claro */
        }
        .stDataFrame, .dataframe {
            background-color: #1A1D23 !important; /* tablas ligeramente más claras */
            color: #FAFAFA !important;
        }
        .stMarkdown, .stText {
            color: #FAFAFA !important;
        }
        .stButton>button {
            background-color: #0077B6;
            color: white;
            border-radius: 8px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #0096C7;
        }
    </style>
""", unsafe_allow_html=True)



@st.cache_data
def cargar_datos():
    vehiculos = pd.read_csv("./working_dir/Vehiculos_2019_2023_diccionario.csv", encoding="utf-8")
    hechos = pd.read_csv("./working_dir/Hechos_2019_2023_diccionario.csv", encoding="utf-8")
    fallecidos = pd.read_csv("./working_dir/Fallecidos_2019_2023_diccionario.csv", encoding="utf-8")
    return vehiculos, hechos, fallecidos

vehiculos, hechos, fallecidos = cargar_datos()

# SIDEBAR
st.sidebar.title("Menu de opciones")
modo = st.sidebar.radio(
    "Selecciona una sección:",
    ("Exploración de Datos", "Modelos de Predicción", "Visualizaciones Detalladas")
)

# SECCIÓN 1: EXPLORACIÓN INTERACTIVA
if modo == "Exploración de Datos":
    st.header("Exploración interactiva de datos")

    dataset = st.selectbox("Selecciona un conjunto de datos:", ["Vehículos", "Hechos", "Fallecidos"])
    df = {"Vehículos": vehiculos, "Hechos": hechos, "Fallecidos": fallecidos}[dataset].copy()

    st.markdown("Filtros globales (enlazan todas las gráficas)")

    col1, col2, col3 = st.columns(3)
    with col1:
        anos = sorted(df["ano_ocu"].dropna().unique()) if "ano_ocu" in df.columns else []
        ano_sel = st.multiselect("Año", anos)
        if ano_sel:
            df = df[df["ano_ocu"].isin(ano_sel)]
    with col2:
        deptos = sorted(df["depto_ocu"].dropna().unique()) if "depto_ocu" in df.columns else []
        depto_sel = st.multiselect("Departamento", deptos)
        if depto_sel:
            df = df[df["depto_ocu"].isin(depto_sel)]
    with col3:
        sexos = sorted(df["sexo_per"].dropna().unique()) if "sexo_per" in df.columns else []
        sexo_sel = st.multiselect("Sexo", sexos)
        if sexo_sel:
            df = df[df["sexo_per"].isin(sexo_sel)]

    st.markdown("Vista previa de los datos filtrados")
    st.dataframe(df.head(20))

    # === Visualizaciones enlazadas ===
    st.subheader("Visualizaciones enlazadas e interactivas")
    vis = st.multiselect("Selecciona las visualizaciones a mostrar:", [
        "Casos por Departamento",
        "Evolución temporal",
        "Top tipos de vehículo",
        "Distribución por Sexo y Edad",
        "Proporción de Tipos de Evento",
        "Casos por Zona",
        "Comparativo Año vs Departamento"
    ], default=["Casos por Departamento", "Top tipos de vehículo", "Distribución por Sexo y Edad"])

    # Casos por Departamento 
    if "Casos por Departamento" in vis and "depto_ocu" in df.columns:
        depto_counts = df["depto_ocu"].value_counts().reset_index()
        depto_counts.columns = ["Departamento", "Casos"]
        fig = px.bar(
            depto_counts,
            x="Departamento",
            y="Casos",
            title="Casos por Departamento",
            color="Casos",
            color_continuous_scale=["#001F3F", "#004080", "#0074D9", "#66B2FF"]
        )
        fig.update_layout(
            xaxis={'categoryorder': 'total descending'},
            title_font=dict(size=20, color="#002B5B", family="Arial Black"),
            yaxis_title="Número de casos",
            xaxis_title="Departamento",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Evolución temporal
    if "Evolución temporal" in vis and "ano_ocu" in df.columns:
        serie = df["ano_ocu"].value_counts().sort_index()
        fig = px.line(serie, markers=True, title="Evolución de casos por año",
                      color_discrete_sequence=["#004080"])
        fig.update_traces(line=dict(width=3))
        st.plotly_chart(fig, use_container_width=True)

    # Top tipos de vehículo + tabla enlazada
    if "Top tipos de vehículo" in vis and "tipo_veh" in df.columns:
        top = df["tipo_veh"].value_counts().head(10)
        fig = px.bar(top, title="Top 10 tipos de vehículo", text=top.values,
                     color_discrete_sequence=["#002B5B"])
        fig.update_traces(marker_line_color="#000000", marker_line_width=1.2)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Registros correspondientes a los Top 10 vehículos (enlace dinámico)**")
        st.dataframe(df[df["tipo_veh"].isin(top.index)].head(50))

    # Distribución Sexo-Edad
    if "Distribución por Sexo y Edad" in vis and "sexo_per" in df.columns and "edad_per" in df.columns:
        fig = px.box(df, x="sexo_per", y="edad_per", title="Distribución de edad por sexo",
                     color="sexo_per", color_discrete_sequence=["#003366", "#FF6600"])
        st.plotly_chart(fig, use_container_width=True)

    # Tipos de evento
    if "Proporción de Tipos de Evento" in vis and "tipo_eve" in df.columns:
        fig = px.pie(df, names="tipo_eve", title="Proporción de tipos de evento",
                     color_discrete_sequence=["#004080", "#0074D9", "#FF851B", "#FF4136", "#2ECC40"])
        st.plotly_chart(fig, use_container_width=True)

    # Casos por Zona
    if "Casos por Zona" in vis and "zona_ocu" in df.columns:
        fig = px.histogram(df, x="zona_ocu", color="zona_ocu",
                           title="Distribución por zona", color_discrete_sequence=["#0074D9"])
        st.plotly_chart(fig, use_container_width=True)

    # Comparativo Año vs Departamento
    if "Comparativo Año vs Departamento" in vis and {"ano_ocu", "depto_ocu"} <= set(df.columns):
        fig = px.density_heatmap(df, x="ano_ocu", y="depto_ocu",
                                 title="Casos por Año y Departamento",
                                 color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

# SECCIÓN 2: MODELOS PREDICTIVOS
elif modo == "Modelos de Predicción":
    st.header("Comparación de Modelos Predictivos")

    df = fallecidos.copy().dropna(subset=["sexo_per"])
    features = ["depto_ocu", "zona_ocu", "tipo_eve", "tipo_veh", "g_hora_5"]
    df = df.dropna(subset=features)

    le = LabelEncoder()
    for col in features + ["sexo_per"]:
        df[col] = le.fit_transform(df[col])

    X = df[features]
    y = df["sexo_per"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    modelos = {
        "Árbol de Decisión": DecisionTreeClassifier(max_depth=6),
        "Random Forest": RandomForestClassifier(n_estimators=200),
        "KNN": KNeighborsClassifier(n_neighbors=8)
    }

    resultados = []
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, pred)
        resultados.append([nombre, acc])

        st.subheader(f"Matriz de Confusión - {nombre}")
        cm = confusion_matrix(y_test, pred)

        # 
        fig, ax = plt.subplots(figsize=(4, 3))  
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            cbar=False,
            annot_kws={"size": 8}  
        )
        ax.set_xlabel("Predicción", fontsize=10)
        ax.set_ylabel("Valor real", fontsize=10)
        ax.tick_params(axis='both', labelsize=8)
        st.pyplot(fig, use_container_width=False)

    comp = pd.DataFrame(resultados, columns=["Modelo", "Precisión"])
    st.subheader("Comparación visual de desempeño")
    modelos_sel = st.multiselect("Selecciona modelos a comparar:", comp["Modelo"], default=comp["Modelo"])
    fig = px.bar(
        comp[comp["Modelo"].isin(modelos_sel)],
        x="Modelo",
        y="Precisión",
        color="Modelo",
        text="Precisión",
        color_discrete_sequence=["#003366", "#FF6600", "#3399FF"]
    )
    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(comp)


# SECCIÓN 3: VISUALIZACIONES DETALLADAS
elif modo == "Visualizaciones Detalladas":
    st.header("Visualizaciones Interactivas Detalladas")
    nivel = st.radio("Nivel de detalle:", ["General", "Por Año", "Por Año y Departamento"])
    df = vehiculos.copy()

    if nivel == "Por Año":
        año = st.selectbox("Año:", sorted(df["ano_ocu"].dropna().unique()))
        df = df[df["ano_ocu"] == año]

    if nivel == "Por Año y Departamento":
        año = st.selectbox("Año:", sorted(df["ano_ocu"].dropna().unique()))
        depto = st.selectbox("Departamento:", sorted(df["depto_ocu"].dropna().unique()))
        df = df[(df["ano_ocu"] == año) & (df["depto_ocu"] == depto)]

    st.subheader("Tipos de vehículos más involucrados")
    top = df["tipo_veh"].value_counts().head(10)
    fig = px.bar(top, title="Top tipos de vehículo más involucrados",
                 color_discrete_sequence=["#002B5B"])
    fig.update_traces(marker_line_color="#000", marker_line_width=1.5)
    st.plotly_chart(fig, use_container_width=True)
