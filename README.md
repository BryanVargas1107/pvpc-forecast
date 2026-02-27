# ⚡ PVPC Forecast — Predicción del Precio de la Luz en España

Proyecto de ciencia de datos para la descarga, análisis y predicción del
**Precio Voluntario para el Pequeño Consumidor (PVPC)** en España, usando
datos públicos de la API de [ESIOS (Red Eléctrica de España)](https://www.esios.ree.es/es/pagina/api).

---

## 🎯 Objetivo

Construir un pipeline completo de ciencia de datos que:
1. **Descargue** automáticamente los precios horarios del PVPC desde la API de REE
2. **Explore y limpie** los datos (EDA)
3. **Modele** la serie temporal con distintos enfoques (estadísticos y ML)
4. **Evalúe y compare** los modelos con métricas estándar

---

## 📁 Estructura del proyecto

```
pvpc-forecast/
├── .env.example        # Plantilla de variables de entorno
├── .gitignore
├── README.md
├── requirements.txt    # Dependencias del proyecto
│
├── data/
│   └── raw/            # Datos originales de la API (no versionados en Git)
│
└── src/
    └── data/
        └── fetch_data.py   # Script de descarga de datos
```

---

## 🚀 Cómo reproducir este proyecto

### 1. Clona el repositorio
```bash
git clone https://github.com/TU_USUARIO/pvpc-forecast.git
cd pvpc-forecast
```

### 2. Crea y activa el entorno virtual
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Instala las dependencias
```bash
pip install -r requirements.txt
```

### 4. Configura tu token de ESIOS
Copia el archivo de ejemplo y añade tu token:
```bash
cp .env.example .env
```
Edita `.env` y sustituye `your_token_here` por tu token real.
Puedes solicitarlo en: https://www.esios.ree.es/es/pagina/api

### 5. Descarga los datos
```bash
python src/data/fetch_data.py
```

---

## 🗺️ Fases del proyecto

| Fase | Descripción | Estado |
|------|-------------|--------|
| 1 | Conexión a la API de ESIOS y descarga de datos | ✅ Completada |
| 2 | Exploración y análisis de datos (EDA) | 🔜 Próximamente |
| 3 | Modelado de la serie temporal | 🔜 Próximamente |
| 4 | Evaluación y comparación de modelos | 🔜 Próximamente |

---

## 🛠️ Tecnologías

- **Python 3.13+**
- `requests` — peticiones HTTP a la API
- `python-dotenv` — gestión segura de credenciales
- *(próximamente)* `pandas`, `matplotlib`, `statsmodels`, `scikit-learn`

---

## 📊 Fuente de datos

- **API:** ESIOS — Sistema de Información del Operador del Sistema (REE)
- **Indicador:** 1001 — PVPC
- **Frecuencia:** Horaria
- **Unidad:** €/MWh

---

## 👤 Autor
Bryan Vargas Sanchez
Proyecto de aprendizaje personal — Grado en Ciencia de Datos