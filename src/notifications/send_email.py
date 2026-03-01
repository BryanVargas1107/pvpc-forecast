"""
send_email.py
-------------
Envía la predicción diaria del PVPC por correo usando SMTP de Gmail.

Configuración necesaria (variables de entorno en .env o GitHub Secrets):
    GMAIL_SENDER    → dirección de Gmail del remitente (ej: pvpc.bot@gmail.com)
    GMAIL_APP_PASS  → contraseña de aplicación de Gmail (16 caracteres)
    EMAIL_RECIPIENTS → destinatarios separados por comas

Cómo obtener la contraseña de aplicación de Gmail:
    1. Activa la verificación en dos pasos en tu cuenta Google
    2. Ve a: myaccount.google.com → Seguridad → Contraseñas de aplicaciones
    3. Crea una contraseña para "Correo" → copia los 16 caracteres
    4. Pégala en .env como GMAIL_APP_PASS (sin espacios)
"""

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


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
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
load_dotenv()

GMAIL_SENDER     = os.getenv("GMAIL_SENDER")
GMAIL_APP_PASS   = os.getenv("GMAIL_APP_PASS")
EMAIL_RECIPIENTS = os.getenv("EMAIL_RECIPIENTS", "")


# ---------------------------------------------------------------------------
# FUNCIONES
# ---------------------------------------------------------------------------

def validate_email_config() -> list[str]:
    """
    Verifica que las variables de entorno necesarias están configuradas.

    Returns:
        Lista de destinatarios validada.

    Raises:
        EnvironmentError: Si falta alguna variable obligatoria.
    """
    if not GMAIL_SENDER:
        raise EnvironmentError("Falta GMAIL_SENDER en .env o GitHub Secrets")
    if not GMAIL_APP_PASS:
        raise EnvironmentError("Falta GMAIL_APP_PASS en .env o GitHub Secrets")
    if not EMAIL_RECIPIENTS:
        raise EnvironmentError("Falta EMAIL_RECIPIENTS en .env o GitHub Secrets")

    recipients = [r.strip() for r in EMAIL_RECIPIENTS.split(",") if r.strip()]
    logger.info(f"Configuración de email válida. Destinatarios: {len(recipients)}")
    return recipients


def build_html_email(df_forecast: pd.DataFrame) -> tuple[str, str]:
    """
    Construye el asunto y el cuerpo HTML del correo con la predicción.

    El correo incluye:
    - Resumen ejecutivo: hora más cara, hora más barata, precio medio
    - Tabla completa con las 24 horas y los intervalos de confianza
    - Código de colores: verde (barato), amarillo (medio), rojo (caro)

    Args:
        df_forecast: DataFrame con columnas hora_local, precio_predicho,
                     limite_inferior, limite_superior.

    Returns:
        Tupla (asunto, cuerpo HTML).
    """
    fecha_hoy    = datetime.now(timezone.utc).strftime("%d/%m/%Y")
    precio_medio = df_forecast["precio_predicho"].mean()
    hora_barata  = df_forecast.loc[df_forecast["precio_predicho"].idxmin(), "hora_local"]
    hora_cara    = df_forecast.loc[df_forecast["precio_predicho"].idxmax(), "hora_local"]
    precio_min   = df_forecast["precio_predicho"].min()
    precio_max   = df_forecast["precio_predicho"].max()

    asunto = f"⚡ Predicción PVPC {fecha_hoy} — Media: {precio_medio:.0f} €/MWh"

    # Construir filas de la tabla con código de colores
    percentil_33 = df_forecast["precio_predicho"].quantile(0.33)
    percentil_66 = df_forecast["precio_predicho"].quantile(0.66)

    filas_html = ""
    for _, row in df_forecast.iterrows():
        precio = row["precio_predicho"]
        if precio <= percentil_33:
            color_fondo = "#d4edda"   # verde claro
            color_texto = "#155724"
        elif precio <= percentil_66:
            color_fondo = "#fff3cd"   # amarillo claro
            color_texto = "#856404"
        else:
            color_fondo = "#f8d7da"   # rojo claro
            color_texto = "#721c24"

        filas_html += f"""
        <tr style="background-color: {color_fondo}; color: {color_texto};">
            <td style="padding: 6px 12px; font-weight: bold;">{row['hora_local']}</td>
            <td style="padding: 6px 12px; text-align: right; font-weight: bold;">{precio:.2f}</td>
            <td style="padding: 6px 12px; text-align: right; color: #666;">
                [{row['limite_inferior']:.0f} – {row['limite_superior']:.0f}]
            </td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: Arial, sans-serif; color: #333; max-width: 600px; margin: 0 auto; }}
            .header {{ background: linear-gradient(135deg, #1a1a2e, #16213e); color: white;
                       padding: 24px; border-radius: 8px 8px 0 0; }}
            .header h1 {{ margin: 0; font-size: 22px; }}
            .header p  {{ margin: 4px 0 0; opacity: 0.8; font-size: 14px; }}
            .summary {{ display: flex; gap: 12px; padding: 16px; background: #f8f9fa;
                        border-left: 4px solid #007bff; margin: 16px 0; border-radius: 4px; }}
            .stat {{ flex: 1; text-align: center; }}
            .stat .valor {{ font-size: 22px; font-weight: bold; color: #007bff; }}
            .stat .etiqueta {{ font-size: 12px; color: #666; margin-top: 4px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
            th {{ background: #343a40; color: white; padding: 10px 12px; text-align: left; }}
            tr:nth-child(even) {{ filter: brightness(0.97); }}
            .footer {{ font-size: 11px; color: #999; text-align: center;
                       padding: 16px; border-top: 1px solid #eee; margin-top: 16px; }}
            .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px;
                      font-size: 11px; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>⚡ Predicción PVPC España</h1>
            <p>Modelo XGBoost Multivariable · {fecha_hoy} · Península</p>
        </div>

        <div style="padding: 16px;">
            <div class="summary">
                <div class="stat">
                    <div class="valor">{precio_medio:.0f} €/MWh</div>
                    <div class="etiqueta">Precio medio</div>
                </div>
                <div class="stat">
                    <div class="valor" style="color: #28a745;">{precio_min:.0f} €/MWh</div>
                    <div class="etiqueta">▼ Mín · {hora_barata[-5:]}</div>
                </div>
                <div class="stat">
                    <div class="valor" style="color: #dc3545;">{precio_max:.0f} €/MWh</div>
                    <div class="etiqueta">▲ Máx · {hora_cara[-5:]}</div>
                </div>
            </div>

            <p style="font-size: 13px; color: #555;">
                💡 <strong>Consejo del día:</strong>
                Programa electrodomésticos de alto consumo (lavadora, lavavajillas,
                cargadores) para las <strong>{hora_barata[-5:]}</strong>,
                cuando el precio es más bajo.
            </p>

            <table>
                <thead>
                    <tr>
                        <th>Hora (CET/CEST)</th>
                        <th style="text-align: right;">Precio (€/MWh)</th>
                        <th style="text-align: right;">Intervalo</th>
                    </tr>
                </thead>
                <tbody>
                    {filas_html}
                </tbody>
            </table>

            <div style="font-size: 12px; color: #666; margin-top: 8px;">
                <span style="background:#d4edda; padding: 2px 8px; border-radius: 4px;">■ Barato</span>
                &nbsp;
                <span style="background:#fff3cd; padding: 2px 8px; border-radius: 4px;">■ Medio</span>
                &nbsp;
                <span style="background:#f8d7da; padding: 2px 8px; border-radius: 4px;">■ Caro</span>
            </div>
        </div>

        <div class="footer">
            Predicción generada automáticamente · Modelo XGBoost (MAE=18.27 €/MWh en test)<br>
            Datos: ESIOS — Red Eléctrica de España · Meteorología: Open-Meteo<br>
            <a href="https://github.com/TU_USUARIO/pvpc-forecast" style="color: #999;">
                Ver código en GitHub
            </a>
        </div>
    </body>
    </html>
    """

    return asunto, html


def send_forecast_email(df_forecast: pd.DataFrame) -> None:
    """
    Envía el correo con la predicción a todos los destinatarios configurados.

    Usa SMTP con TLS sobre el puerto 587 de Gmail.
    Requiere una contraseña de aplicación (no la contraseña normal de Gmail).

    Args:
        df_forecast: DataFrame con las predicciones generadas por predict.py.
    """
    recipients = validate_email_config()
    asunto, html = build_html_email(df_forecast)

    # Construir mensaje
    msg = MIMEMultipart("alternative")
    msg["Subject"] = asunto
    msg["From"]    = f"PVPC Bot ⚡ <{GMAIL_SENDER}>"
    msg["To"]      = ", ".join(recipients)

    msg.attach(MIMEText(html, "html", "utf-8"))

    # Enviar via SMTP con TLS
    logger.info(f"Conectando a smtp.gmail.com:587...")
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.ehlo()
        server.starttls()
        server.login(GMAIL_SENDER, GMAIL_APP_PASS)
        server.sendmail(GMAIL_SENDER, recipients, msg.as_string())

    logger.info(f"✅ Correo enviado a {len(recipients)} destinatario(s)")


if __name__ == "__main__":
    # Test rápido con datos ficticios para verificar la configuración
    import pandas as pd
    from datetime import datetime, timedelta, timezone

    print("Probando configuración de email...")
    base = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    df_test = pd.DataFrame([
        {
            "hora_local":      (base + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M"),
            "precio_predicho": 80 + i * 5,
            "limite_inferior": 70 + i * 5,
            "limite_superior": 90 + i * 5,
        }
        for i in range(24)
    ])

    send_forecast_email(df_test)
