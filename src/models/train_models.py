"""
train_models.py
---------------
Pipeline de modelado para la predicción del PVPC.

Modelos implementados:
    1. Baseline Naive    — referencia mínima obligatoria
    2. SARIMA            — modelo estadístico clásico para series estacionales
    3. Prophet           — modelo de Meta para series con múltiples estacionalidades

Flujo:
    CSV limpio → split train/val/test → estacionariedad → modelos → evaluación
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")  # Prophet y statsmodels son verbosos

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
DATA_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR         = Path(__file__).resolve().parents[2] / "models"

# Proporciones de la división temporal
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15  (el resto)


# ---------------------------------------------------------------------------
# 1. CARGA DE DATOS
# ---------------------------------------------------------------------------

def load_processed_data() -> pd.Series:
    """
    Carga el CSV procesado más reciente y devuelve una Serie temporal.

    Devuelve una pd.Series (no DataFrame) porque los modelos de series
    temporales esperan una secuencia unidimensional con índice temporal.

    Returns:
        Serie con índice datetime UTC y valores de precio en €/MWh.
    """
    csv_files = sorted(
        DATA_PROCESSED_DIR.glob("pvpc_clean_*.csv"),
        key=lambda f: f.stat().st_mtime,
    )
    if not csv_files:
        raise FileNotFoundError(
            "No se encontró ningún CSV en data/processed/. "
            "Ejecuta primero process_data.py."
        )

    latest = csv_files[-1]
    logger.info(f"Cargando datos procesados: {latest.name}")

    df = pd.read_csv(latest, index_col="datetime_utc", parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)

    serie = df["precio_eur_mwh"].sort_index()
    logger.info(f"Serie cargada: {len(serie)} registros | {serie.index.min()} → {serie.index.max()}")
    return serie


# ---------------------------------------------------------------------------
# 2. DIVISIÓN TRAIN / VALIDATION / TEST
# ---------------------------------------------------------------------------

def split_data(serie: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Divide la serie en tres bloques cronológicos: train, validation y test.

    IMPORTANTE: la división es siempre cronológica, nunca aleatoria.
    Mezclar fechas futuras en el entrenamiento causaría data leakage,
    haciendo que las métricas sean irrealmente buenas.

    Args:
        serie: Serie temporal completa.

    Returns:
        Tupla (train, val, test) como pd.Series.
    """
    n = len(serie)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train = serie.iloc[:n_train]
    val   = serie.iloc[n_train : n_train + n_val]
    test  = serie.iloc[n_train + n_val :]

    logger.info(f"Train: {len(train)} registros | {train.index.min().date()} → {train.index.max().date()}")
    logger.info(f"Val:   {len(val)}   registros | {val.index.min().date()} → {val.index.max().date()}")
    logger.info(f"Test:  {len(test)}  registros | {test.index.min().date()} → {test.index.max().date()}")

    return train, val, test


# ---------------------------------------------------------------------------
# 3. TEST DE ESTACIONARIEDAD (ADF)
# ---------------------------------------------------------------------------

def test_stationarity(serie: pd.Series) -> dict:
    """
    Aplica el test de Dickey-Fuller Aumentado (ADF) para comprobar
    si la serie es estacionaria.

    Hipótesis:
        H0 (nula):      la serie NO es estacionaria (tiene raíz unitaria)
        H1 (alternativa): la serie SÍ es estacionaria

    Interpretación del p-valor:
        p < 0.05 → rechazamos H0 → la serie ES estacionaria
        p ≥ 0.05 → no podemos rechazar H0 → la serie NO es estacionaria

    Una serie estacionaria tiene media y varianza constantes en el tiempo,
    condición que SARIMA requiere (o que se consigue diferenciando).

    Args:
        serie: Serie temporal a analizar.

    Returns:
        Diccionario con estadístico, p-valor y conclusión.
    """
    from statsmodels.tsa.stattools import adfuller

    resultado = adfuller(serie.dropna(), autolag="AIC")

    es_estacionaria = resultado[1] < 0.05
    conclusion = "ESTACIONARIA ✓" if es_estacionaria else "NO estacionaria ✗"

    logger.info(f"Test ADF — Estadístico: {resultado[0]:.4f} | p-valor: {resultado[1]:.4f} → {conclusion}")

    return {
        "estadistico_adf": round(resultado[0], 4),
        "p_valor":         round(resultado[1], 4),
        "valores_criticos": resultado[4],
        "es_estacionaria": es_estacionaria,
        "conclusion":      conclusion,
    }


# ---------------------------------------------------------------------------
# 4. MÉTRICAS DE EVALUACIÓN
# ---------------------------------------------------------------------------

def compute_metrics(y_real: pd.Series, y_pred: pd.Series, nombre_modelo: str) -> dict:
    """
    Calcula MAE, RMSE y MAPE para comparar modelos con una misma vara.

    - MAE  (Mean Absolute Error):       error medio en €/MWh. Fácil de interpretar.
    - RMSE (Root Mean Squared Error):   penaliza más los errores grandes. Sensible a picos.
    - MAPE (Mean Absolute % Error):     error porcentual. Permite comparar entre series distintas.

    Args:
        y_real:        Valores reales del período de evaluación.
        y_pred:        Predicciones del modelo.
        nombre_modelo: Nombre para identificar el modelo en el reporte.

    Returns:
        Diccionario con las tres métricas.
    """
    # Alinear índices por si tienen timestamps distintos
    y_real, y_pred = y_real.align(y_pred, join="inner")

    mae  = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100

    logger.info(f"[{nombre_modelo}] MAE={mae:.2f} | RMSE={rmse:.2f} | MAPE={mape:.2f}%")

    return {
        "modelo": nombre_modelo,
        "MAE":    round(mae, 2),
        "RMSE":   round(rmse, 2),
        "MAPE":   round(mape, 2),
    }


# ---------------------------------------------------------------------------
# 5. MODELO 1 — BASELINE NAIVE (SEASONAL)
# ---------------------------------------------------------------------------

def naive_seasonal_forecast(train: pd.Series, horizonte: int, periodo: int = 24) -> pd.Series:
    """
    Modelo baseline: predice copiando el valor de hace exactamente
    un ciclo completo (por defecto 24 horas = un día).

    Es el modelo de referencia obligatorio. Si un modelo más complejo
    no supera al naive, no aporta valor real.

    Ejemplo: para predecir el precio a las 20:00 del martes,
             usa el precio a las 20:00 del lunes.

    Args:
        train:     Serie de entrenamiento.
        horizonte: Número de pasos a predecir.
        periodo:   Longitud del ciclo estacional (24 = diario).

    Returns:
        Serie con las predicciones naive.
    """
    logger.info(f"Entrenando Naive Seasonal (periodo={periodo}h, horizonte={horizonte}h)...")

    # Cogemos los últimos 'periodo' valores del train y los repetimos
    ultimos_ciclo = train.iloc[-periodo:]
    repeticiones  = (horizonte // periodo) + 1
    predicciones  = pd.concat([ultimos_ciclo] * repeticiones).iloc[:horizonte]

    # Asignar los timestamps correctos del futuro
    frecuencia   = pd.tseries.frequencies.to_offset(
        pd.infer_freq(train.index[-periodo*2:])
    )
    indice_futuro = pd.date_range(
        start=train.index[-1] + frecuencia,
        periods=horizonte,
        freq=frecuencia,
        tz=train.index.tz,
    )
    predicciones.index = indice_futuro

    logger.info("Naive Seasonal — predicciones generadas ✓")
    return predicciones


# ---------------------------------------------------------------------------
# 6. MODELO 2 — SARIMA
# ---------------------------------------------------------------------------

def fit_sarima(train: pd.Series, val: pd.Series) -> tuple:
    """
    Ajusta un modelo SARIMA sobre los datos de entrenamiento y genera
    predicciones para el período de validación.

    SARIMA(p,d,q)(P,D,Q,s) tiene dos partes:
        (p,d,q) — componente no estacional:
            p = orden AR (autorregresivo): influencia de valores pasados
            d = diferenciación: veces que diferenciamos para estacionarizar
            q = orden MA (media móvil): influencia de errores pasados

        (P,D,Q,s) — componente estacional con periodo s:
            S=24 para capturar el patrón diario

    Parámetros elegidos con balance entre rendimiento y coste computacional.
    Para producción se usaría auto_arima (librería pmdarima) para buscar
    automáticamente los mejores parámetros.

    Args:
        train: Serie de entrenamiento.
        val:   Serie de validación (define el horizonte de predicción).

    Returns:
        Tupla (modelo ajustado, predicciones como pd.Series).
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    logger.info("Entrenando SARIMA(1,1,1)(1,1,1,24) — esto puede tardar unos minutos...")

    # Trabajamos con la serie sin zona horaria para compatibilidad con statsmodels
    train_values = train.copy()
    train_values.index = train_values.index.tz_localize(None)

    modelo = SARIMAX(
        train_values,
        order=(1, 1, 1),            # (p, d, q)
        seasonal_order=(1, 1, 1, 24),  # (P, D, Q, s=24h)
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    resultado = modelo.fit(disp=False)
    logger.info(f"SARIMA ajustado — AIC: {resultado.aic:.2f}")

    # Predicciones para el horizonte de validación
    predicciones_raw = resultado.forecast(steps=len(val))
    predicciones = pd.Series(
        predicciones_raw.values,
        index=val.index,
        name="sarima",
    )

    return resultado, predicciones


# ---------------------------------------------------------------------------
# 7. MODELO 3 — PROPHET
# ---------------------------------------------------------------------------

def fit_prophet(train: pd.Series, val: pd.Series) -> tuple:
    """
    Ajusta un modelo Prophet sobre los datos de entrenamiento.

    Prophet descompone la serie en:
        - Tendencia (trend):         cambios graduales a largo plazo
        - Estacionalidad anual:      patrones que se repiten cada año
        - Estacionalidad semanal:    patrones de lunes a domingo
        - Estacionalidad diaria:     el ciclo de 24 horas
        - Festivos (opcional):       días que rompen el patrón habitual

    Prophet espera un DataFrame con columnas exactas 'ds' (fecha) y 'y' (valor).

    Args:
        train: Serie de entrenamiento.
        val:   Serie de validación.

    Returns:
        Tupla (modelo ajustado, predicciones como pd.Series).
    """
    from prophet import Prophet

    logger.info("Entrenando Prophet...")

    # Prophet requiere un DataFrame con columnas 'ds' y 'y'
    # Además no acepta zona horaria en el índice, la eliminamos
    df_prophet = pd.DataFrame({
        "ds": train.index.tz_localize(None),
        "y":  train.values,
    })

    modelo = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False,   # ← desactivar: solo tenemos 7 meses
    changepoint_prior_scale=0.1,    # ← más flexible que antes (0.05)
    seasonality_prior_scale=5,      # ← reducimos para evitar overfitting
    seasonality_mode="multiplicative",  # ← mejor para series con picos altos
)

    modelo.fit(df_prophet)
    logger.info("Prophet ajustado ✓")

    # Generar predicciones para el horizonte de validación
    futuro = pd.DataFrame({"ds": val.index.tz_localize(None)})
    forecast = modelo.predict(futuro)

    predicciones = pd.Series(
        forecast["yhat"].values,
        index=val.index,
        name="prophet",
    )

    # Prophet puede predecir valores negativos — el precio nunca es negativo
    # (en la práctica el PVPC tiene un suelo regulado)
    predicciones = predicciones.clip(lower=0)

    return modelo, predicciones, forecast


# ---------------------------------------------------------------------------
# 8. TABLA COMPARATIVA
# ---------------------------------------------------------------------------

def compare_models(resultados: list[dict]) -> pd.DataFrame:
    """
    Construye una tabla comparativa con las métricas de todos los modelos.

    Args:
        resultados: Lista de diccionarios devueltos por compute_metrics().

    Returns:
        DataFrame ordenado por MAE (de mejor a peor).
    """
    df_comparativa = pd.DataFrame(resultados).set_index("modelo")
    df_comparativa = df_comparativa.sort_values("MAE")

    # Añadir columna de ranking
    df_comparativa.insert(0, "ranking", range(1, len(df_comparativa) + 1))

    logger.info("\n" + "="*45)
    logger.info("TABLA COMPARATIVA DE MODELOS")
    logger.info("="*45)
    logger.info("\n" + df_comparativa.to_string())

    return df_comparativa


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    """
    Ejecuta el pipeline completo de modelado y muestra los resultados.
    """
    # 1. Cargar datos
    serie = load_processed_data()

    # 2. Dividir
    train, val, test = split_data(serie)

    # 3. Test de estacionariedad (sobre train únicamente — no miramos el futuro)
    print("\n" + "="*45)
    print("TEST DE ESTACIONARIEDAD (ADF)")
    print("="*45)
    resultado_adf = test_stationarity(train)
    print(f"  Estadístico ADF : {resultado_adf['estadistico_adf']}")
    print(f"  p-valor         : {resultado_adf['p_valor']}")
    print(f"  Conclusión      : {resultado_adf['conclusion']}")

    # 4. Modelos — evaluamos sobre validación
    metricas = []

    # --- Naive ---
    pred_naive = naive_seasonal_forecast(train, horizonte=len(val))
    metricas.append(compute_metrics(val, pred_naive, "Naive Seasonal"))

    # --- SARIMA ---
    _, pred_sarima = fit_sarima(train, val)
    metricas.append(compute_metrics(val, pred_sarima, "SARIMA(1,1,1)(1,1,1,24)"))

    # --- Prophet ---
    _, pred_prophet, _ = fit_prophet(train, val)
    metricas.append(compute_metrics(val, pred_prophet, "Prophet"))

    # 5. Tabla comparativa
    print("\n")
    tabla = compare_models(metricas)
    print(tabla.to_string())

    mejor_modelo = tabla.index[0]
    print(f"\n🏆 Mejor modelo en validación: {mejor_modelo}")
    print("\n⚠️  El conjunto de TEST no se ha usado todavía.")
    print("    Úsalo solo una vez, cuando hayas elegido el modelo definitivo.")


if __name__ == "__main__":
    main()
