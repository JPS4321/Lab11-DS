import streamlit as st
import pandas as pd
from municipios_diccionario import mapa_mupio

# ======================
# ======================
st.set_page_config(
    page_title="Cuadro de mando de accidentes 2019–2023",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# CARGA DE DATOS Y DICCIONARIOS
# ======================
@st.cache_data
def cargar_datos():
    vehiculos = pd.read_csv("./working_dir/Vehiculos_2019_2023.csv")
    hechos = pd.read_csv("./working_dir/Hechos_2019_2023.csv")
    fallecidos = pd.read_csv("./working_dir/Fallecidos_2019_2023.csv")

    # =========================
    # DICCIONARIOS DE CÓDIGOS
    # =========================

    mapa_hora_ocu = {99: "Ignorado"}
    mapa_g_hora = {
        1: "00:00 a 05:59",
        2: "06:00 a 11:59",
        3: "12:00 a 17:59",
        4: "18:00 a 23:59"
    }

    mapa_g_hora_5 = {1: "Mañana", 2: "Tarde", 3: "Noche"}

    mapa_mes = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo",
        6: "Junio", 7: "Julio", 8: "Agosto", 9: "Septiembre",
        10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }

    mapa_dia_sem = {
        1: "Lunes", 2: "Martes", 3: "Miércoles", 4: "Jueves",
        5: "Viernes", 6: "Sábado", 7: "Domingo"
    }

    mapa_zona = {1: "Urbana", 2: "Rural", 99: "Ignorada"}
    mapa_sexo = {1: "Hombre", 2: "Mujer", 9: "Ignorado"}

    mapa_edad = {
        1: "Menor de 15", 2: "15-19", 3: "20-24", 4: "25-29",
        5: "30-34", 6: "35-39", 7: "40-44", 8: "45-49",
        9: "50-54", 10: "55-59", 11: "60-64", 12: "65-69",
        13: "70-74", 14: "75-80", 15: "80 y más", 999: "Ignorada"
    }

    mapa_g_edad_80 = {
        1: "Menor de 15", 2: "15-19", 3: "20-24", 4: "25-29",
        5: "30-34", 6: "35-39", 7: "40-44", 8: "45-49",
        9: "50-54", 10: "55-59", 11: "60-64", 12: "65-69",
        13: "70-74", 14: "75-80", 15: "80 y más", 16: "Ignorado"
    }

    mapa_g_edad_60 = {
        1: "Menor de 15", 2: "15-19", 3: "20-24", 4: "25-29",
        5: "30-34", 6: "35-39", 7: "40-44", 8: "45-49",
        9: "50-54", 10: "55-59", 11: "60 y más", 12: "Ignorado"
    }

    mapa_quinquenal = {
        1: "0-4", 2: "5-9", 3: "10-14", 4: "15-19", 5: "20-24", 6: "25-29",
        7: "30-34", 8: "35-39", 9: "40-44", 10: "45-49", 11: "50-54",
        12: "55-59", 13: "60-64", 14: "65-69", 15: "70-74",
        16: "75-79", 17: "80 o más", 18: "Ignorado"
    }

    mapa_estado_con = {1: "No ebrio", 2: "Ebrio", 9: "Ignorado"}
    mapa_mayor_menor = {1: "Mayor", 2: "Menor", 9: "Ignorado"}

    mapa_depto = {
        1: "Guatemala", 2: "El Progreso", 3: "Sacatepéquez", 4: "Chimaltenango",
        5: "Escuintla", 6: "Santa Rosa", 7: "Sololá", 8: "Totonicapán",
        9: "Quetzaltenango", 10: "Suchitepéquez", 11: "Retalhuleu",
        12: "San Marcos", 13: "Huehuetenango", 14: "Quiché",
        15: "Baja Verapaz", 16: "Alta Verapaz", 17: "Petén",
        18: "Izabal", 19: "Zacapa", 20: "Chiquimula",
        21: "Jalapa", 22: "Jutiapa"
    }

    mapa_tipo_veh = {
        1: "Automóvil", 2: "Camioneta sport o blazer", 3: "Pick up", 4: "Motocicleta",
        5: "Camión", 6: "Cabezal", 7: "Bus extraurbano", 8: "Jeep", 9: "Microbús",
        10: "Taxi", 11: "Panel", 12: "Bus urbano", 13: "Tractor", 14: "Moto taxi",
        15: "Furgón", 16: "Grúa", 17: "Bus escolar", 18: "Bicicleta",
        19: "Avioneta", 20: "Montacargas", 21: "Bus militar", 22: "Cuatrimoto",
        23: "Furgoneta", 99: "Ignorado"
    }

    mapa_color = {
        1: "Rojo", 2: "Blanco", 3: "Azul", 4: "Gris", 5: "Negro", 6: "Verde",
        7: "Amarillo", 8: "Celeste", 9: "Corinto", 10: "Café", 11: "Beige",
        12: "Turquesa", 13: "Marfil", 14: "Anaranjado", 15: "Morado",
        16: "Rosado", 17: "Varios colores", 99: "Ignorado"
    }

    mapa_modelo = {9999: "Ignorado"}

    mapa_g_modelo = {
        1: "1970-1979", 2: "1980-1989", 3: "1990-1999",
        4: "2000-2009", 5: "2010-2019", 99: "Ignorado"
    }

    mapa_tipo_eve = {
        1: "Colisión", 2: "Choque", 3: "Vuelco", 4: "Caída", 5: "Atropello",
        6: "Derrape", 7: "Embarranco", 8: "Encuentro", 99: "Ignorado"
    }

    mapa_fall_les = {1: "Fallecido", 2: "Lesionado", 9: "Ignorado"}
    mapa_internado = {1: "Internado", 2: "No internado", 9: "Ignorado"}

    # =========================
    # REEMPLAZOS AUTOMÁTICOS
    # =========================
    reemplazos = {
        "hora_ocu": mapa_hora_ocu,
        "g_hora": mapa_g_hora,
        "g_hora_5": mapa_g_hora_5,
        "mes_ocu": mapa_mes,
        "dia_sem_ocu": mapa_dia_sem,
        "zona_ocu": mapa_zona,
        "sexo_per": mapa_sexo,
        "edad_per": mapa_edad,
        "g_edad_80ymás": mapa_g_edad_80,
        "g_edad_60ymás": mapa_g_edad_60,
        "edad_quinquenales": mapa_quinquenal,
        "estado_con": mapa_estado_con,
        "mayor_menor": mapa_mayor_menor,
        "depto_ocu": mapa_depto,
        "mupio_ocu": mapa_mupio,
        "tipo_veh": mapa_tipo_veh,
        "color_veh": mapa_color,
        "modelo_veh": mapa_modelo,
        "g_modelo_veh": mapa_g_modelo,
        "tipo_eve": mapa_tipo_eve,
        "fall_les": mapa_fall_les,
        "int_o_noint": mapa_internado
    }

    for df in [vehiculos, hechos, fallecidos]:
        df.replace(["nan", "NaN", "None", "NULL", ""], None, inplace=True)
        for col, mapa in reemplazos.items():
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace({str(k): v for k, v in mapa.items()})
                df[col] = df[col].replace(mapa)
                df[col] = df[col].replace({"99": "Ignorado", "999": "Ignorado", "0": "Ignorado"})

    return vehiculos, hechos, fallecidos


vehiculos, hechos, fallecidos = cargar_datos()

# ======================
# SIDEBAR Y EXPLORACIÓN
# ======================
st.sidebar.title("📊 Cuadro de mando interactivo")
modo = st.sidebar.radio("Selecciona una sección:", ("Exploración de Datos",))

if modo == "Exploración de Datos":
    st.header("🔍 Exploración interactiva de datos")

    dataset = st.selectbox("Selecciona un conjunto de datos:", ["Vehículos", "Hechos", "Fallecidos"])
    df = {"Vehículos": vehiculos, "Hechos": hechos, "Fallecidos": fallecidos}[dataset]

    st.markdown("### 🎚️ Filtros de visualización")

    if "año_ocu" in df.columns:
        años = sorted(df["año_ocu"].dropna().unique())
        año_sel = st.selectbox("Selecciona año:", [None] + list(años))
        if año_sel:
            df = df[df["año_ocu"] == año_sel]

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
