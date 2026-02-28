"""
predict.py
----------
Genera predicciones del PVPC para las próximas 24 horas usando Prophet.

Uso:
    python predict.py              # predice las próximas 24 horas
    python predict.py --horas 48   # predice las próximas 48 horas

Flujo:
    API ESIOS → datos frescos → limpieza → Prophet → predicción → CSV
"""

import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# Importamos las funciones ya construidas en fases anteriores
# Así no duplicamos código — reutilizamos lo que ya funciona
import sys
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from data.fetch_data    import fetch_pvpc, save_raw_json
from data.process_data  import load_raw_json, filter_by_geo, clean_data
from models.train_models import fit_prophet


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
PREDICTIONS_DIR  = Path(__file__).resolve().parent / "data" / "predictions"
DAYS_OF_HISTORY  = 60   # días de histórico para entrenar Prophet en producción


# ---------------------------------------------------------------------------
# FUNCIONES
# ---------------------------------------------------------------------------

def fetch_recent_data() -> pd.Series:
    """
    Descarga los últimos DAYS_OF_HISTORY días frescos de la API y
    devuelve una serie temporal limpia lista para entrenar.

    Siempre descarga datos frescos para que el modelo conozca
    los patrones de precios más recientes.
    """
    today      = datetime.now(timezone.utc).date()
    start_date = (today - timedelta(days=DAYS_OF_HISTORY)).strftime("%Y-%m-%d")
    end_date   = today.strftime("%Y-%m-%d")

    logger.info(f"Descargando datos frescos: {start_date} → {end_date}")
    raw_json = fetch_pvpc(start_date=start_date, end_date=end_date)

    filename = f"pvpc_{start_date}_{end_date}.json"
    save_raw_json(raw_json, filename)

    df_raw      = load_raw_json(Path(__file__).resolve().parent / "data" / "raw" / filename)
    df_filtered = filter_by_geo(df_raw)
    serie       = clean_data(df_filtered)["precio_eur_mwh"]

    logger.info(f"Datos listos: {len(serie)} horas de histórico")
    return serie


def generate_forecast(serie: pd.Series, horas: int) -> pd.DataFrame:
    """
    Entrena Prophet sobre la serie recibida y genera predicciones.

    Para predicción en producción entrenamos con todos los datos
    disponibles (no dividimos en train/val/test — eso es solo
    para evaluar, no para predecir en producción).

    Args:
        serie: Serie temporal limpia con histórico reciente.
        horas: Número de horas a predecir hacia el futuro.

    Returns:
        DataFrame con columnas: datetime_utc, hora_local,
        precio_predicho, limite_inferior, limite_superior.
    """
    logger.info(f"Entrenando Prophet sobre {len(serie)} horas de datos...")

    # Creamos un conjunto de validación ficticio del tamaño del horizonte
    # solo para que fit_prophet pueda generar el número correcto de pasos
    frecuencia    = pd.tseries.frequencies.to_offset(pd.infer_freq(serie.index[-48:]))
    indice_futuro = pd.date_range(
        start=serie.index[-1] + frecuencia,
        periods=horas,
        freq=frecuencia,
        tz=serie.index.tz,
    )
    horizonte_ficticio = pd.Series(index=indice_futuro, dtype=float)

    _, pred_serie, forecast_completo = fit_prophet(serie, horizonte_ficticio)

    # Enriquecer con intervalos de confianza y hora local española
    df_result = pd.DataFrame({
        "datetime_utc":      pred_serie.index,
        "precio_predicho":   forecast_completo["yhat"].values.clip(min=0).round(2),
        "limite_inferior":   forecast_completo["yhat_lower"].values.clip(min=0).round(2),
        "limite_superior":   forecast_completo["yhat_upper"].values.clip(min=0).round(2),
    })

    # Añadir hora local española (CET/CEST) para legibilidad
    df_result["hora_local"] = (
        df_result["datetime_utc"]
        .dt.tz_convert("Europe/Madrid")
        .dt.strftime("%Y-%m-%d %H:%M")
    )

    return df_result


def save_predictions(df: pd.DataFrame) -> Path:
    """
    Guarda las predicciones en data/predictions/ con timestamp en el nombre.

    Usar un directorio separado para predicciones evita mezclarlas
    con los datos procesados históricos.
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    filename  = f"prediccion_pvpc_{timestamp}.csv"
    filepath  = PREDICTIONS_DIR / filename

    df.to_csv(filepath, index=False)
    logger.info(f"Predicciones guardadas en: {filepath}")
    return filepath


def print_forecast_table(df: pd.DataFrame) -> None:
    """
    Imprime las predicciones en consola con formato legible.
    Destaca las horas más caras y más baratas.
    """
    precio_max = df["precio_predicho"].max()
    precio_min = df["precio_predicho"].min()

    print("\n" + "=" * 65)
    print("  PREDICCIÓN PVPC — PRÓXIMAS HORAS (Península)")
    print("=" * 65)
    print(f"  {'Hora local (CET/CEST)':<25} {'Precio (€/MWh)':>14}  {'Intervalo':>20}")
    print("-" * 65)

    for _, row in df.iterrows():
        precio   = row["precio_predicho"]
        intervalo = f"[{row['limite_inferior']:.0f} – {row['limite_superior']:.0f}]"

        # Etiqueta visual para máximo y mínimo
        if precio == precio_max:
            etiqueta = "  ▲ más caro"
        elif precio == precio_min:
            etiqueta = "  ▼ más barato"
        else:
            etiqueta = ""

        print(f"  {row['hora_local']:<25} {precio:>12.2f}  {intervalo:>22}{etiqueta}")

    print("=" * 65)
    print(f"  Media predicha:  {df['precio_predicho'].mean():.2f} €/MWh")
    print(f"  Hora más barata: {df.loc[df['precio_predicho'].idxmin(), 'hora_local']}")
    print(f"  Hora más cara:   {df.loc[df['precio_predicho'].idxmax(), 'hora_local']}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predice el precio del PVPC usando Prophet."
    )
    parser.add_argument(
        "--horas",
        type=int,
        default=24,
        help="Número de horas a predecir (default: 24)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    logger.info(f"Iniciando predicción para las próximas {args.horas} horas")

    # 1. Datos frescos de la API
    serie = fetch_recent_data()

    # 2. Generar predicción
    df_forecast = generate_forecast(serie, horas=args.horas)

    # 3. Mostrar en consola
    print_forecast_table(df_forecast)

    # 4. Guardar CSV
    ruta = save_predictions(df_forecast)
    print(f"\n📁 CSV guardado en: {ruta}")


if __name__ == "__main__":
    main()
