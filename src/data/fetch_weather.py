"""
fetch_weather.py
----------------
Descarga datos meteorológicos históricos horarios desde Open-Meteo.

Open-Meteo es una API gratuita, sin registro y sin token.
Documentación: https://open-meteo.com/en/docs/historical-weather-api

Ubicación usada: Madrid (40.42°N, 3.70°W)
Justificación: Madrid es el nodo de referencia del mercado eléctrico
español y su climatología representa bien el comportamiento de la demanda
peninsular (temperaturas extremas en verano e invierno).

Variables descargadas y su relación con el precio eléctrico:
    - temperature_2m       → demanda de calefacción/refrigeración
    - cloudcover           → producción solar fotovoltaica (↑ nubes = ↓ solar)
    - shortwave_radiation  → producción solar directa (W/m²)
    - windspeed_10m        → producción eólica (↑ viento = ↓ precio)
    - precipitation        → producción hidroeléctrica y demanda doméstica
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests


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
OPEN_METEO_URL   = "https://archive-api.open-meteo.com/v1/archive"
DATA_RAW_DIR     = Path(__file__).resolve().parents[2] / "data" / "raw"

# Coordenadas de Madrid
MADRID_LAT = 40.42
MADRID_LON = -3.70

# Variables meteorológicas a descargar
WEATHER_VARIABLES = [
    "temperature_2m",       # temperatura (°C)
    "cloudcover",           # nubosidad (%)
    "shortwave_radiation",  # radiación solar directa (W/m²)
    "windspeed_10m",        # velocidad del viento a 10m (km/h)
    "precipitation",        # precipitación (mm)
]


# ---------------------------------------------------------------------------
# FUNCIONES
# ---------------------------------------------------------------------------

def fetch_weather(start_date: str, end_date: str) -> dict:
    """
    Descarga datos meteorológicos históricos horarios de Open-Meteo.

    La API devuelve datos en formato JSON con una clave 'hourly' que
    contiene listas paralelas: una lista de timestamps y una lista
    por cada variable meteorológica.

    Args:
        start_date: Fecha de inicio en formato 'YYYY-MM-DD'.
        end_date:   Fecha de fin en formato 'YYYY-MM-DD'.

    Returns:
        Diccionario JSON completo con la respuesta de la API.

    Raises:
        requests.HTTPError: Si la API devuelve un error HTTP.
    """
    params = {
        "latitude":        MADRID_LAT,
        "longitude":       MADRID_LON,
        "start_date":      start_date,
        "end_date":        end_date,
        "hourly":          ",".join(WEATHER_VARIABLES),
        "timezone":        "UTC",
        "wind_speed_unit": "kmh",
    }

    logger.info(f"Solicitando datos meteorológicos {start_date} → {end_date}...")
    response = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    n_horas = len(data.get("hourly", {}).get("time", []))
    logger.info(f"Respuesta recibida: {n_horas} horas de datos meteorológicos")

    return data


def save_weather_json(data: dict, filename: str) -> Path:
    """
    Guarda el JSON meteorológico en data/raw/.

    Args:
        data:     JSON completo de Open-Meteo.
        filename: Nombre del archivo de salida.

    Returns:
        Ruta completa del archivo guardado.
    """
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_RAW_DIR / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"JSON meteorológico guardado en: {output_path}")
    return output_path


def main():
    """
    Descarga datos meteorológicos para el mismo período que el PVPC.
    Por defecto: últimos 60 días (igual que predict.py).
    Acepta --start y --end como argumentos de línea de comandos.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Descarga datos meteorológicos históricos de Open-Meteo."
    )
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end",   type=str, default=None)
    args = parser.parse_args()

    today      = datetime.now(timezone.utc).date()
    end_date   = args.end   or today.strftime("%Y-%m-%d")
    start_date = args.start or (today - timedelta(days=60)).strftime("%Y-%m-%d")

    data     = fetch_weather(start_date, end_date)
    filename = f"weather_{start_date}_{end_date}.json"
    save_weather_json(data, filename)

    # Muestra una muestra de los datos
    hourly = data.get("hourly", {})
    print("\n--- Primeros 3 registros ---")
    for i in range(min(3, len(hourly.get("time", [])))):
        print(f"  {hourly['time'][i]}")
        for var in WEATHER_VARIABLES:
            print(f"    {var}: {hourly[var][i]}")


if __name__ == "__main__":
    main()
