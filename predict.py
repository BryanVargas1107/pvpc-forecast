"""
predict.py
----------
Genera predicciones del PVPC para las próximas 24 horas usando XGBoost
entrenado sobre datos de precio + variables meteorológicas de Open-Meteo.

Uso:
    python predict.py              # predice y muestra en consola
    python predict.py --email      # predice y envía correo a los destinatarios
    python predict.py --horas 48   # predice las próximas 48 horas

Flujo:
    API ESIOS → precio fresco
    API Open-Meteo → meteorología fresca + próximas horas (forecast)
    Fusión + features → XGBoost → predicción → consola / email
"""

import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from data.fetch_data    import fetch_pvpc, save_raw_json
from data.fetch_weather import fetch_weather, save_weather_json
from data.process_data  import (
    load_raw_json, filter_by_geo, clean_data,
    load_weather_json, merge_price_weather, build_features,
)
from models.train_models import fit_xgboost, FEATURE_COLS


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
PREDICTIONS_DIR = Path(__file__).resolve().parent / "data" / "predictions"
DAYS_OF_HISTORY = 60


# ---------------------------------------------------------------------------
# FUNCIONES
# ---------------------------------------------------------------------------

def fetch_fresh_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Descarga precio PVPC y meteorología para el rango dado,
    los fusiona y construye las features para XGBoost.

    Returns:
        DataFrame con todas las features listo para predecir.
    """
    # --- Precio ---
    logger.info("Descargando precio PVPC...")
    raw_pvpc = fetch_pvpc(start_date=start_date, end_date=end_date)
    pvpc_file = f"pvpc_{start_date}_{end_date}.json"
    save_raw_json(raw_pvpc, pvpc_file)

    df_raw      = load_raw_json(Path("data/raw") / pvpc_file)
    df_filtered = filter_by_geo(df_raw)
    df_precio   = clean_data(df_filtered)

    # --- Meteorología histórica ---
    logger.info("Descargando meteorología histórica...")
    raw_weather = fetch_weather(start_date, end_date)
    weather_file = f"weather_{start_date}_{end_date}.json"
    save_weather_json(raw_weather, weather_file)
    df_weather = load_weather_json(Path("data/raw") / weather_file)

    # --- Fusión y features ---
    df_merged   = merge_price_weather(df_precio, df_weather)
    df_features = build_features(df_merged)

    return df_features


def fetch_weather_forecast(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Descarga el pronóstico meteorológico de Open-Meteo para las próximas horas.
    """
    import requests

    params = {
        "latitude":        40.42,
        "longitude":       -3.70,
        "start_date":      start_date,
        "end_date":        end_date,
        "hourly":          "temperature_2m,cloudcover,shortwave_radiation,windspeed_10m,precipitation",
        "timezone":        "UTC",
        "wind_speed_unit": "kmh",
    }

    logger.info(f"Descargando forecast meteorológico {start_date} → {end_date}...")
    response = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=30)
    response.raise_for_status()

    data   = response.json()
    hourly = data["hourly"]

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()

    logger.info(f"Forecast meteorológico: {len(df)} horas")
    return df


def generate_xgboost_forecast(df_history: pd.DataFrame, horas: int) -> pd.DataFrame:
    """
    Entrena XGBoost sobre el histórico reciente y predice las próximas horas
    de forma recursiva, usando forecast meteorológico real de Open-Meteo.
    """
    # Entrenar con todo el histórico (últimas 24h como val para early stopping)
    val_size   = 24
    train_data = df_history.iloc[:-val_size]
    val_data   = df_history.iloc[-val_size:]
    modelo, _  = fit_xgboost(train_data, val_data)

    # Forecast meteorológico para las horas futuras
    today    = datetime.now(timezone.utc).date()
    tomorrow = today + timedelta(days=max(2, horas // 24 + 1))
    df_meteo_forecast = fetch_weather_forecast(
        today.strftime("%Y-%m-%d"),
        tomorrow.strftime("%Y-%m-%d"),
    )

    freq            = pd.tseries.frequencies.to_offset("h")
    ultima_hora     = df_history.index[-1]
    precio_conocido = df_history["precio_eur_mwh"].copy()

    resultados = []
    for i in range(1, horas + 1):
        hora_pred = ultima_hora + i * freq

        # Features temporales
        features = {
            "hora":             hora_pred.hour,
            "dia_semana":       hora_pred.dayofweek,
            "mes":              hora_pred.month,
            "es_fin_de_semana": int(hora_pred.dayofweek >= 5),
        }

        # Lags — precio real o predicho según disponibilidad
        def get_precio(ts):
            if ts in precio_conocido.index:
                return precio_conocido[ts]
            for r in resultados:
                if r["datetime_utc"] == ts:
                    return r["precio_predicho"]
            return precio_conocido.iloc[-1]

        features["precio_lag_24h"]   = get_precio(hora_pred - timedelta(hours=24))
        features["precio_lag_48h"]   = get_precio(hora_pred - timedelta(hours=48))
        features["precio_lag_168h"]  = get_precio(hora_pred - timedelta(hours=168))
        features["precio_media_24h"] = precio_conocido.iloc[-24:].mean()

        # Meteorología del forecast
        if hora_pred in df_meteo_forecast.index:
            meteo = df_meteo_forecast.loc[hora_pred]
            features["temperature_2m"]      = meteo["temperature_2m"]
            features["cloudcover"]          = meteo["cloudcover"]
            features["shortwave_radiation"] = meteo["shortwave_radiation"]
            features["windspeed_10m"]       = meteo["windspeed_10m"]
            features["precipitation"]       = meteo["precipitation"]
        else:
            misma_hora = df_history[df_history.index.hour == hora_pred.hour]
            for col in ["temperature_2m", "cloudcover", "shortwave_radiation",
                        "windspeed_10m", "precipitation"]:
                features[col] = misma_hora[col].mean() if col in misma_hora.columns else 0

        # Predicción
        X           = pd.DataFrame([features])[FEATURE_COLS]
        precio_pred = max(0, float(modelo.predict(X)[0]))
        margen      = precio_pred * 0.15
        hora_local  = hora_pred.tz_convert("Europe/Madrid").strftime("%Y-%m-%d %H:%M")

        resultados.append({
            "datetime_utc":    hora_pred,
            "hora_local":      hora_local,
            "precio_predicho": round(precio_pred, 2),
            "limite_inferior": round(max(0, precio_pred - margen), 2),
            "limite_superior": round(precio_pred + margen, 2),
        })

    return pd.DataFrame(resultados)


def save_predictions(df: pd.DataFrame) -> Path:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    filepath  = PREDICTIONS_DIR / f"prediccion_pvpc_{timestamp}.csv"
    df.to_csv(filepath, index=False)
    logger.info(f"Predicciones guardadas en: {filepath}")
    return filepath


def print_forecast_table(df: pd.DataFrame) -> None:
    precio_max = df["precio_predicho"].max()
    precio_min = df["precio_predicho"].min()

    print("\n" + "=" * 68)
    print("  PREDICCIÓN PVPC — PRÓXIMAS HORAS (Península) · XGBoost")
    print("=" * 68)
    print(f"  {'Hora local (CET/CEST)':<25} {'Precio (€/MWh)':>14}  {'Intervalo':>22}")
    print("-" * 68)

    for _, row in df.iterrows():
        precio    = row["precio_predicho"]
        intervalo = f"[{row['limite_inferior']:.0f} – {row['limite_superior']:.0f}]"
        etiqueta  = "  ▲ más caro" if precio == precio_max else ("  ▼ más barato" if precio == precio_min else "")
        print(f"  {row['hora_local']:<25} {precio:>12.2f}  {intervalo:>24}{etiqueta}")

    print("=" * 68)
    print(f"  Media predicha:  {df['precio_predicho'].mean():.2f} €/MWh")
    print(f"  Hora más barata: {df.loc[df['precio_predicho'].idxmin(), 'hora_local']}")
    print(f"  Hora más cara:   {df.loc[df['precio_predicho'].idxmax(), 'hora_local']}")
    print("=" * 68)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predice el precio del PVPC con XGBoost + datos meteorológicos."
    )
    parser.add_argument("--horas", type=int, default=24)
    parser.add_argument("--email", action="store_true",
                        help="Enviar predicción por correo electrónico")
    return parser.parse_args()


def main():
    args = parse_arguments()
    logger.info(f"Iniciando predicción XGBoost para las próximas {args.horas} horas")

    today      = datetime.now(timezone.utc).date()
    start_date = (today - timedelta(days=DAYS_OF_HISTORY)).strftime("%Y-%m-%d")
    end_date   = today.strftime("%Y-%m-%d")

    df_history  = fetch_fresh_data(start_date, end_date)
    df_forecast = generate_xgboost_forecast(df_history, horas=args.horas)

    print_forecast_table(df_forecast)

    ruta = save_predictions(df_forecast)
    print(f"\n📁 CSV guardado en: {ruta}")

    if args.email:
        from notifications.send_email import send_forecast_email
        send_forecast_email(df_forecast)


if __name__ == "__main__":
    main()
