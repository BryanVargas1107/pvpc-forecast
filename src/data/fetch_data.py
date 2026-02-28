"""
fetch_data.py
-------------
Descarga los datos del PVPC (Precio Voluntario para el Pequeño Consumidor)
desde la API pública de ESIOS (Red Eléctrica de España).

Indicador usado: 1001 — PVPC (€/MWh)
Documentación API: https://www.esios.ree.es/es/pagina/api
"""

import argparse
import os
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from pprint import pprint

import requests
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# CONFIGURACIÓN DEL LOGGING
# El logging es mejor que los print(): guarda un registro de lo que pasa,
# con nivel de severidad (INFO, WARNING, ERROR) y marca de tiempo.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CONSTANTES
# Agrupar las constantes aquí arriba facilita cambiarlas sin tocar la lógica.
# ---------------------------------------------------------------------------
ESIOS_BASE_URL = "https://api.esios.ree.es"
PVPC_INDICATOR_ID = 1001          # ID del indicador PVPC en ESIOS
DATA_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


# ---------------------------------------------------------------------------
# CARGA DE VARIABLES DE ENTORNOPYTHON 
# load_dotenv() lee el archivo .env y mete las variables en el entorno.
# Así el token nunca aparece escrito en el código.
# ---------------------------------------------------------------------------
load_dotenv()


def get_esios_token() -> str:
    """
    Lee el token de autenticación desde las variables de entorno.

    Returns:
        El token como string.

    Raises:
        EnvironmentError: Si el token no está definido en el .env
    """
    token = os.getenv("ESIOS_TOKEN")
    if not token:
        raise EnvironmentError(
            "No se encontró ESIOS_TOKEN. "
            "Crea un archivo .env con tu token (consulta .env.example)."
        )
    return token


def build_headers(token: str) -> dict:
    """
    Construye las cabeceras HTTP que requiere la API de ESIOS.

    Cabeceras requeridas:
    - Authorization: autenticación con token personal de ESIOS.
    - Accept / Content-Type: indican que trabajamos con JSON.
    - User-Agent: identifica el cliente. Sin esta cabecera el firewall
      de REE rechaza la petición con un 403 Forbidden.
    - x-api-key: cabecera alternativa de autenticación, añadida por
      compatibilidad con distintas versiones del gateway de la API.
    """
    return {
        "Accept": "application/json; application/vnd.esios-api-v2+json",
        "Content-Type": "application/json",
        "Authorization": f"Token token={token}",
        "x-api-key": token,  # Añadimos esta clave por compatibilidad
        "User-Agent": "pvpc-forecast-app/1.0 (Estudiante Data Science)" # Nuestro "disfraz"
    }


def fetch_pvpc(start_date: str, end_date: str) -> dict:
    """
    Hace la petición GET a la API de ESIOS y devuelve el JSON completo.

    La URL sigue el patrón:
    GET /indicators/{id}?start_date=...&end_date=...&time_trunc=hour

    Args:
        start_date: Fecha de inicio en formato 'YYYY-MM-DD'.
        end_date:   Fecha de fin en formato 'YYYY-MM-DD'.

    Returns:
        Diccionario Python con la respuesta JSON completa de la API.

    Raises:
        requests.HTTPError: Si la API devuelve un código de error HTTP.
    """
    token = get_esios_token()
    headers = build_headers(token)

    # La API espera las fechas en formato ISO 8601 con hora y zona horaria
    url = f"{ESIOS_BASE_URL}/indicators/{PVPC_INDICATOR_ID}"
    params = {
        "start_date": f"{start_date}T00:00:00Z",
        "end_date": f"{end_date}T23:59:59Z",
        "time_trunc": "hour",   # agrupación horaria
    }

    logger.info(f"Solicitando PVPC del {start_date} al {end_date}...")

    response = requests.get(url, headers=headers, params=params, timeout=30)

    # Lanza una excepción automáticamente si el código HTTP es 4xx o 5xx
    response.raise_for_status()

    logger.info(f"Respuesta recibida. Código HTTP: {response.status_code}")
    return response.json()


def parse_pvpc(raw_json: dict) -> list[dict]:
    """
    Extrae del JSON solo los campos que nos interesan: hora y precio.

    La estructura del JSON de ESIOS es:
    {
      "indicator": {
        "values": [
          { "datetime": "2024-01-01T01:00:00.000+01:00", "value": 85.23, ... },
          ...
        ]
      }
    }

    Args:
        raw_json: El JSON completo devuelto por la API.

    Returns:
        Lista de diccionarios con claves 'datetime' y 'precio_eur_mwh'.
    """
    try:
        values = raw_json["indicator"]["values"]

    except KeyError as e:
        raise ValueError(f"Estructura del JSON inesperada. Clave no encontrada: {e}")

    records = []
    for entry in values:
        records.append(
            {
                "datetime": entry["datetime"],
                "precio_eur_mwh": entry["value"],
            }
        )

    logger.info(f"Registros extraídos: {len(records)} horas de datos.")
    return records


def save_raw_json(raw_json: dict, filename: str) -> Path:
    """
    Guarda el JSON original en data/raw/ sin modificaciones.

    Es buena práctica guardar siempre los datos en crudo.
    Si más adelante cambias el procesamiento, tienes los originales.

    Args:
        raw_json: El JSON completo de la API.
        filename: Nombre del archivo de salida (sin ruta).

    Returns:
        La ruta completa donde se guardó el archivo.
    """
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_RAW_DIR / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(raw_json, f, ensure_ascii=False, indent=2)

    logger.info(f"JSON original guardado en: {output_path}")
    return output_path

def parse_arguments() -> argparse.Namespace:
    """
    Define y procesa los argumentos de línea de comandos.

    Permite especificar el rango de fechas al ejecutar el script,
    lo que facilita descargar históricos sin modificar el código.

    Uso:
        # Últimos 7 días (por defecto)
        python src/data/fetch_data.py

        # Rango personalizado
        python src/data/fetch_data.py --start 2025-08-01 --end 2026-02-27
    """
    parser = argparse.ArgumentParser(
        description="Descarga datos del PVPC desde la API de ESIOS (REE)."
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Fecha de inicio en formato YYYY-MM-DD (default: hace 7 días)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Fecha de fin en formato YYYY-MM-DD (default: hoy)",
    )
    return parser.parse_args()

def main():
    """
    Punto de entrada principal del script.

    Por defecto descarga los últimos 7 días. Acepta --start y --end
    como argumentos para descargar cualquier rango histórico.
    """
    args = parse_arguments()

    today = datetime.now(timezone.utc).date()

    end_date   = args.end   or today.strftime("%Y-%m-%d")
    start_date = args.start or (today - timedelta(days=7)).strftime("%Y-%m-%d")

    # 1. Descarga
    raw_json = fetch_pvpc(start_date=start_date, end_date=end_date)

    # 2. Guarda el JSON original
    filename = f"pvpc_{start_date}_{end_date}.json"
    save_raw_json(raw_json, filename)

    # 3. Extrae los campos de interés
    records = parse_pvpc(raw_json)

    # 4. Muestra una muestra en consola para verificar que todo funciona
    print("\n--- Primeros 5 registros ---")
    for record in records[:5]:
        print(f"  {record['datetime']}  →  {record['precio_eur_mwh']:.2f} €/MWh")

    print(f"\n✅ Total de registros descargados: {len(records)}")
    print(f"📁 Datos guardados en: data/raw/{filename}")


if __name__ == "__main__":
    main()