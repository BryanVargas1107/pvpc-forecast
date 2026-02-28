"""
process_data.py
---------------
Limpieza, transformación y análisis exploratorio del PVPC.

Responsabilidad única: tomar el JSON crudo descargado por fetch_data.py
y devolver un DataFrame limpio y listo para modelar.

Flujo:
    JSON crudo → DataFrame → filtrar zona → limpiar → validar → guardar
"""

import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CONSTANTES
# ---------------------------------------------------------------------------
GEO_ID_PENINSULA = 8741
DATA_RAW_DIR      = Path(__file__).resolve().parents[2] / "data" / "raw"
DATA_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


# ---------------------------------------------------------------------------
# 1. CARGA
# ---------------------------------------------------------------------------

def load_raw_json(filepath: str | Path) -> pd.DataFrame:
    """
    Lee el JSON crudo de ESIOS y lo convierte en un DataFrame.

    Mantiene todas las zonas geográficas intactas — el filtrado
    se hace en una función separada para no perder datos.

    Args:
        filepath: Ruta al archivo JSON descargado por fetch_data.py.

    Returns:
        DataFrame con todas las filas y columnas originales del JSON.
    """
    filepath = Path(filepath)
    logger.info(f"Cargando archivo: {filepath.name}")

    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    df = pd.DataFrame(raw["indicator"]["values"])
    logger.info(f"DataFrame cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
    logger.info(f"Zonas disponibles: {df['geo_name'].unique().tolist()}")
    return df


# ---------------------------------------------------------------------------
# 2. FILTRADO
# ---------------------------------------------------------------------------

def filter_by_geo(df: pd.DataFrame, geo_id: int = GEO_ID_PENINSULA) -> pd.DataFrame:
    """
    Filtra el DataFrame por zona geográfica.

    Args:
        df:     DataFrame completo con todas las zonas.
        geo_id: ID de la zona a conservar (default: Península = 8741).

    Returns:
        DataFrame filtrado con solo la zona indicada.
    """
    zone_name = df.loc[df["geo_id"] == geo_id, "geo_name"].iloc[0]
    filtered = df[df["geo_id"] == geo_id].copy()
    logger.info(f"Zona seleccionada: {zone_name} → {len(filtered)} registros")
    return filtered


# ---------------------------------------------------------------------------
# 3. LIMPIEZA
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica todas las transformaciones de limpieza al DataFrame filtrado.

    Pasos:
    1. Seleccionar y renombrar solo las columnas necesarias
    2. Convertir datetime_utc a tipo datetime (de texto a fecha real)
    3. Establecer datetime como índice (obligatorio para series temporales)
    4. Ordenar cronológicamente
    5. Detectar y eliminar duplicados
    6. Detectar valores nulos
    7. Verificar que no hay huecos temporales en la serie

    Args:
        df: DataFrame filtrado por zona geográfica.

    Returns:
        DataFrame limpio con índice temporal y columna 'precio_eur_mwh'.
    """
    # --- Paso 1: columnas útiles ---
    df = df[["datetime_utc", "geo_name", "value"]].copy()
    df = df.rename(columns={"value": "precio_eur_mwh"})

    # --- Paso 2: convertir a datetime ---
    # utc=True indica que la zona horaria es UTC, evitando ambigüedades
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)

    # --- Paso 3: establecer como índice ---
    df = df.set_index("datetime_utc")

    # --- Paso 4: ordenar cronológicamente ---
    df = df.sort_index()

    # --- Paso 5: duplicados ---
    n_duplicados = df.index.duplicated().sum()
    if n_duplicados > 0:
        logger.warning(f"Se encontraron {n_duplicados} timestamps duplicados → eliminados")
        df = df[~df.index.duplicated(keep="first")]
    else:
        logger.info("Duplicados: ninguno ✓")

    # --- Paso 6: valores nulos ---
    n_nulos = df["precio_eur_mwh"].isnull().sum()
    if n_nulos > 0:
        logger.warning(f"Valores nulos encontrados: {n_nulos}")
        # Interpolación lineal: rellena el hueco con la media entre vecinos
        df["precio_eur_mwh"] = df["precio_eur_mwh"].interpolate(method="time")
        logger.info("Nulos rellenados con interpolación lineal")
    else:
        logger.info("Valores nulos: ninguno ✓")

    # --- Paso 7: huecos temporales ---
    # Generamos el rango completo de horas que debería existir
    rango_completo = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="h",
        tz="UTC",
    )
    horas_faltantes = rango_completo.difference(df.index)
    if len(horas_faltantes) > 0:
        logger.warning(f"Horas faltantes en la serie: {len(horas_faltantes)}")
        logger.warning(f"Primera hora faltante: {horas_faltantes[0]}")
    else:
        logger.info("Continuidad temporal: serie completa sin huecos ✓")

    logger.info(f"DataFrame limpio: {df.shape[0]} registros desde {df.index.min()} hasta {df.index.max()}")
    return df


# ---------------------------------------------------------------------------
# 4. ANÁLISIS EXPLORATORIO
# ---------------------------------------------------------------------------

def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estadísticas descriptivas del precio y añade métricas extra.

    Args:
        df: DataFrame limpio con índice temporal.

    Returns:
        DataFrame con las estadísticas.
    """
    stats = df["precio_eur_mwh"].describe()

    # Métricas extra que describe() no incluye
    stats["mediana"]   = df["precio_eur_mwh"].median()
    stats["rango"]     = stats["max"] - stats["min"]
    stats["coef_var"]  = (stats["std"] / stats["mean"]) * 100  # variabilidad relativa en %
    stats["asimetria"] = df["precio_eur_mwh"].skew()           # >0 cola derecha, <0 cola izquierda

    return stats.round(2)


def compute_hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el precio medio, máximo y mínimo por hora del día (0-23).

    Útil para identificar las franjas horarias más y menos caras.

    Returns:
        DataFrame con índice 0-23 y columnas mean, max, min.
    """
    profile = df.groupby(df.index.hour)["precio_eur_mwh"].agg(["mean", "max", "min"])
    profile.index.name = "hora"
    return profile.round(2)


def compute_daily_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el precio medio por día de la semana (0=lunes, 6=domingo).

    Returns:
        DataFrame con el precio medio por día de la semana.
    """
    day_names = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    profile = df.groupby(df.index.dayofweek)["precio_eur_mwh"].mean()
    profile.index = day_names
    profile.name = "precio_medio_eur_mwh"
    return profile.round(2)


# ---------------------------------------------------------------------------
# 5. DETECCIÓN DE ANOMALÍAS (método IQR)
# ---------------------------------------------------------------------------

def detect_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """
    Detecta outliers usando el método del Rango Intercuartílico (IQR).

    El IQR es la distancia entre el percentil 25 y el 75 (la "caja" del
    boxplot). Se considera outlier cualquier valor que quede fuera del
    rango [Q1 - factor*IQR, Q3 + factor*IQR].

    Con factor=1.5 detectamos outliers moderados (estándar de Tukey).
    Con factor=3.0 detectamos solo outliers extremos.

    Args:
        df:     DataFrame limpio.
        factor: Multiplicador del IQR. Default 1.5 (convención estadística).

    Returns:
        DataFrame con solo las filas identificadas como outliers,
        con columnas extra que explican por qué son outliers.
    """
    q1  = df["precio_eur_mwh"].quantile(0.25)
    q3  = df["precio_eur_mwh"].quantile(0.75)
    iqr = q3 - q1

    limite_inferior = q1 - factor * iqr
    limite_superior = q3 + factor * iqr

    logger.info(f"IQR: Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
    logger.info(f"Límites: [{limite_inferior:.2f}, {limite_superior:.2f}] €/MWh")

    mascara_outliers = (
        (df["precio_eur_mwh"] < limite_inferior) |
        (df["precio_eur_mwh"] > limite_superior)
    )

    outliers = df[mascara_outliers].copy()
    outliers["tipo"] = np.where(
        outliers["precio_eur_mwh"] > limite_superior, "alto", "bajo"
    )
    outliers["desviacion_iqr"] = (
        (outliers["precio_eur_mwh"] - df["precio_eur_mwh"].median()) / iqr
    ).round(2)

    logger.info(f"Outliers detectados: {len(outliers)} ({len(outliers)/len(df)*100:.1f}% del total)")
    return outliers


# ---------------------------------------------------------------------------
# 6. GUARDADO
# ---------------------------------------------------------------------------

def save_processed(df: pd.DataFrame, filename: str) -> Path:
    """
    Guarda el DataFrame procesado como CSV en data/processed/.

    Usamos CSV (no JSON) porque es el formato estándar para
    DataFrames tabulares y pandas lo lee muy eficientemente.

    Args:
        df:       DataFrame limpio y procesado.
        filename: Nombre del archivo de salida.

    Returns:
        Ruta completa del archivo guardado.
    """
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_PROCESSED_DIR / filename

    df.to_csv(output_path)
    logger.info(f"CSV procesado guardado en: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# MAIN — ejecuta el pipeline completo
# ---------------------------------------------------------------------------

def main():
    """
    Pipeline completo: carga → filtra → limpia → analiza → guarda.
    """
    # Busca el JSON más reciente en data/raw/
    raw_files = sorted(DATA_RAW_DIR.glob("pvpc_*.json"))
    if not raw_files:
        raise FileNotFoundError(
            "No se encontró ningún archivo JSON en data/raw/. "
            "Ejecuta primero fetch_data.py."
        )
    latest_file = raw_files[-1]
    logger.info(f"Archivo más reciente: {latest_file.name}")

    # Pipeline
    df_raw       = load_raw_json(latest_file)
    df_peninsula = filter_by_geo(df_raw, geo_id=GEO_ID_PENINSULA)
    df_clean     = clean_data(df_peninsula)

    # Estadísticas en consola
    print("\n" + "="*50)
    print("ESTADÍSTICAS DESCRIPTIVAS (€/MWh)")
    print("="*50)
    print(compute_descriptive_stats(df_clean).to_string())

    print("\n" + "="*50)
    print("PRECIO MEDIO POR HORA DEL DÍA")
    print("="*50)
    print(compute_hourly_profile(df_clean).to_string())

    print("\n" + "="*50)
    print("PRECIO MEDIO POR DÍA DE LA SEMANA")
    print("="*50)
    print(compute_daily_profile(df_clean).to_string())

    # Outliers
    outliers = detect_outliers_iqr(df_clean)
    if not outliers.empty:
        print("\n" + "="*50)
        print("ANOMALÍAS DETECTADAS (método IQR)")
        print("="*50)
        print(outliers[["precio_eur_mwh", "tipo", "desviacion_iqr"]].to_string())

    # Guardar
    output_name = latest_file.stem.replace("pvpc_", "pvpc_clean_") + ".csv"
    save_processed(df_clean, output_name)

    print(f"\n✅ Procesamiento completado → data/processed/{output_name}")


# ---------------------------------------------------------------------------
# 7. CARGA Y FUSIÓN DE DATOS METEOROLÓGICOS
# ---------------------------------------------------------------------------

def load_weather_json(filepath: str | Path) -> pd.DataFrame:
    """
    Lee el JSON de Open-Meteo y lo convierte en un DataFrame horario.

    Open-Meteo devuelve listas paralelas bajo la clave 'hourly':
    una lista de timestamps y una lista por cada variable meteorológica.
    Las convertimos en columnas de un DataFrame con índice temporal UTC.

    Args:
        filepath: Ruta al archivo JSON descargado por fetch_weather.py.

    Returns:
        DataFrame con índice datetime UTC y una columna por variable.
    """
    filepath = Path(filepath)
    logger.info(f"Cargando datos meteorológicos: {filepath.name}")

    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    hourly = raw["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()

    logger.info(f"Datos meteorológicos cargados: {df.shape[0]} horas × {df.shape[1]} variables")
    return df


def merge_price_weather(
    df_price: pd.DataFrame,
    df_weather: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fusiona el DataFrame de precios con el meteorológico por timestamp UTC.

    Usamos un join de tipo 'inner' para quedarnos solo con las horas
    que tienen datos en ambos DataFrames. Esto evita introducir NaN
    en las variables meteorológicas, que confundirían al modelo.

    Args:
        df_price:   DataFrame limpio del PVPC con índice datetime UTC.
        df_weather: DataFrame meteorológico con índice datetime UTC.

    Returns:
        DataFrame combinado con precio y variables meteorológicas.
    """
    # Eliminamos la columna 'geo_name' que no es útil para el modelo
    df_price_clean = df_price[["precio_eur_mwh"]].copy()

    df_merged = df_price_clean.join(df_weather, how="inner")

    n_dropped = len(df_price_clean) - len(df_merged)
    if n_dropped > 0:
        logger.warning(f"Horas eliminadas por falta de datos meteorológicos: {n_dropped}")

    logger.info(
        f"Dataset combinado: {df_merged.shape[0]} horas × {df_merged.shape[1]} variables "
        f"({df_merged.index.min().date()} → {df_merged.index.max().date()})"
    )
    return df_merged


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye variables de entrada (features) para el modelo XGBoost.

    XGBoost no entiende fechas — hay que convertir el tiempo en números
    que capture los patrones temporales. Añadimos también variables lag
    (el precio de horas anteriores), que son muy predictivas en series
    temporales eléctricas.

    Features temporales:
        hora, dia_semana, mes, es_fin_de_semana

    Features lag (precio pasado):
        precio_lag_24h   → precio de ayer a la misma hora
        precio_lag_48h   → precio de hace 2 días a la misma hora
        precio_lag_168h  → precio de hace 7 días a la misma hora (mismo día semana)
        precio_media_24h → media de las últimas 24 horas (tendencia reciente)

    Features meteorológicas:
        temperature_2m, cloudcover, shortwave_radiation,
        windspeed_10m, precipitation

    Args:
        df: DataFrame combinado precio + meteorología con índice UTC.

    Returns:
        DataFrame con todas las features y la variable objetivo,
        sin filas NaN (las primeras 168 horas se pierden por los lags).
    """
    df = df.copy()

    # --- Features temporales ---
    df["hora"]            = df.index.hour
    df["dia_semana"]      = df.index.dayofweek      # 0=lunes, 6=domingo
    df["mes"]             = df.index.month
    df["es_fin_de_semana"] = (df.index.dayofweek >= 5).astype(int)

    # --- Features lag ---
    df["precio_lag_24h"]   = df["precio_eur_mwh"].shift(24)
    df["precio_lag_48h"]   = df["precio_eur_mwh"].shift(48)
    df["precio_lag_168h"]  = df["precio_eur_mwh"].shift(168)  # mismo día, semana pasada
    df["precio_media_24h"] = df["precio_eur_mwh"].rolling(24).mean()

    # Eliminar filas con NaN (los primeros 168 registros no tienen lag completo)
    n_antes = len(df)
    df = df.dropna()
    logger.info(f"Features construidas. Filas eliminadas por lags: {n_antes - len(df)} | Quedan: {len(df)}")

    return df


def save_multivariate_dataset(df: pd.DataFrame, filename: str) -> Path:
    """
    Guarda el dataset multivariable en data/processed/.

    Args:
        df:       DataFrame con features y variable objetivo.
        filename: Nombre del archivo de salida.

    Returns:
        Ruta completa del archivo guardado.
    """
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_PROCESSED_DIR / filename
    df.to_csv(output_path)
    logger.info(f"Dataset multivariable guardado en: {output_path}")
    return output_path


if __name__ == "__main__":
    main()
