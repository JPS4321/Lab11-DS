import pandas as pd
from municipios_diccionario import mapa_mupio  # Importa tu diccionario de municipios

# ============================================================
# FUNCION GENERAL PARA REEMPLAZAR LOS CODIGOS POR SUS TEXTOS
# ============================================================

def reemplazar_codigos(df, diccionarios):
    for col, mapping in diccionarios.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: mapping.get(x, mapping.get(int(x), x))
                                    if pd.notna(x) and (str(x).isdigit() or x in mapping) else x)
    return df

# ============================================================
# DICCIONARIOS COMUNES
# ============================================================

diccionarios_comunes = {
    "hora_ocu": {99: "Ignorado"},
    "g_hora": {
        1: "00:00 a 05:59", 2: "06:00 a 11:59", 3: "12:00 a 17:59", 4: "18:00 a 23:59"
    },
    "g_hora_5": {1: "Manana", 2: "Tarde", 3: "Noche"},
    "mes_ocu": {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
        7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    },
    "dia_sem_ocu": {
        1: "Lunes", 2: "Martes", 3: "Miercoles", 4: "Jueves", 5: "Viernes", 6: "Sabado", 7: "Domingo"
    },
    "depto_ocu": {
        1: "Guatemala", 2: "El Progreso", 3: "Sacatepequez", 4: "Chimaltenango", 5: "Escuintla",
        6: "Santa Rosa", 7: "Solola", 8: "Totonicapan", 9: "Quetzaltenango", 10: "Suchitepequez",
        11: "Retalhuleu", 12: "San Marcos", 13: "Huehuetenango", 14: "Quiche", 15: "Baja Verapaz",
        16: "Alta Verapaz", 17: "Peten", 18: "Izabal", 19: "Zacapa", 20: "Chiquimula",
        21: "Jalapa", 22: "Jutiapa"
    },
    "mupio_ocu": mapa_mupio,
    "zona_ocu": {99: "Ignorada"},
    "sexo_per": {1: "Hombre", 2: "Mujer", 9: "Ignorado"},
    "edad_per": {999: "Ignorada"},
    "g_edad_80ymas": {
        1: "Menor de 15", 2: "15-19", 3: "20-24", 4: "25-29", 5: "30-34", 6: "35-39",
        7: "40-44", 8: "45-49", 9: "50-54", 10: "55-59", 11: "60-64", 12: "65-69",
        13: "70-74", 14: "75-80", 15: "80 y mas", 16: "Ignorado"
    },
    "g_edad_60ymas": {
        1: "Menor de 15", 2: "15-19", 3: "20-24", 4: "25-29", 5: "30-34", 6: "35-39",
        7: "40-44", 8: "45-49", 9: "50-54", 10: "55-59", 11: "60 y mas", 12: "Ignorado"
    },
    "edad_quinquenales": {
        1: "0 - 4", 2: "5 - 9", 3: "10 - 14", 4: "15 - 19", 5: "20 - 24", 6: "25 - 29",
        7: "30 - 34", 8: "35 - 39", 9: "40 - 44", 10: "45 - 49", 11: "50 - 54",
        12: "55 - 59", 13: "60 - 64", 14: "65 - 69", 15: "70 - 74", 16: "75 - 79",
        17: "80 o mas", 18: "Ignorado"
    },
    "estado_con": {1: "No ebrio", 2: "Ebrio", 9: "Ignorado"},
    "mayor_menor": {1: "Mayor", 2: "Menor", 9: "Ignorado"},
    "tipo_veh": {
        1: "Automovil", 2: "Camioneta sport o blazer", 3: "Pick up", 4: "Motocicleta", 5: "Camion",
        6: "Cabezal", 7: "Bus extraurbano", 8: "Jeep", 9: "Microbus", 10: "Taxi", 11: "Panel",
        12: "Bus urbano", 13: "Tractor", 14: "Moto taxi", 15: "Furgon", 16: "Grua", 17: "Bus escolar",
        18: "Bicicleta", 19: "Avioneta", 20: "Montacargas", 21: "Bus militar", 22: "Cuatrimoto",
        23: "Furgoneta", 99: "Ignorado"
    },
    "color_veh": {
        1: "Rojo", 2: "Blanco", 3: "Azul", 4: "Gris", 5: "Negro", 6: "Verde", 7: "Amarillo",
        8: "Celeste", 9: "Corinto", 10: "Cafe", 11: "Beige", 12: "Turquesa", 13: "Marfil",
        14: "Anaranjado", 15: "Morado", 16: "Rosado", 17: "Varios colores", 99: "Ignorado"
    },
    "g_modelo_veh": {
        1: "1970-1979", 2: "1980-1989", 3: "1990-1999", 4: "2000-2009", 5: "2010-2019", 99: "Ignorado"
    },
    "tipo_eve": {
        1: "Colision", 2: "Choque", 3: "Vuelco", 4: "Caida", 5: "Atropello",
        6: "Derrape", 7: "Embarranco", 8: "Encunetamiento", 99: "Ignorado"
    },
    "fall_les": {1: "Fallecido", 2: "Lesionado"},
    "int_o_noint": {1: "Internado", 2: "No internado", 9: "Ignorado"}
}

# ============================================================
# ARCHIVOS A PROCESAR
# ============================================================

archivos = [
    "./working_dir/Vehiculos_2019_2023.csv",
    "./working_dir/Hechos_2019_2023.csv",
    "./working_dir/Fallecidos_2019_2023.csv"
]

# ============================================================
# PROCESAMIENTO
# ============================================================

for archivo in archivos:
    print(f"\nProcesando: {archivo}")
    df = pd.read_csv(archivo, encoding='utf-8-sig')  # Soporte UTF-8 con BOM
    df = reemplazar_codigos(df, diccionarios_comunes)

    # 🔹 Eliminar columnas innecesarias
    columnas_a_eliminar = ['modelo_veh', 'marca_veh']
    df = df.drop(columns=[col for col in columnas_a_eliminar if col in df.columns], errors='ignore')

    salida = archivo.replace(".csv", "_diccionario.csv")
    df.to_csv(salida, index=False, encoding='utf-8', lineterminator='\n')
    print(f"✅ Guardado: {salida}")

print("\nTodos los archivos fueron procesados correctamente.")
