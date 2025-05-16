# Bitcoin Trading Analysis

Este proyecto proporciona herramientas para analizar y realizar backtesting de estrategias de trading de Bitcoin utilizando datos históricos de Binance.

## Características

- Descarga de datos históricos de precios desde Binance
- Cálculo de indicadores técnicos (RSI, MACD, EMAs, Bollinger Bands)
- Generación de señales de trading basadas en indicadores técnicos
- Backtesting de estrategias con stop loss y take profit
- Modelo predictivo basado en Random Forest
- Visualización de resultados

## Requisitos

- Python 3.8+
- Cuenta de Binance con API key y secret

## Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/yourusername/bitcoin-trading-analysis.git
cd bitcoin-trading-analysis
```

2. Crea un entorno virtual e instala las dependencias:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Crea un archivo `.env` en la raíz del proyecto con tus credenciales de Binance:
```
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
```

## Estructura del Proyecto

```
.
├── data/
│   ├── fetch_prices.py      # Descarga de precios históricos
│   └── update_data.py       # Actualización de datos existentes
├── features/
│   └── feature_engineering.py  # Cálculo de indicadores técnicos
├── models/
│   ├── backtesting.py      # Simulación de estrategias
│   └── prediction.py       # Modelo predictivo
├── visualization/
│   └── plotting.py         # Funciones de visualización
├── main.py                 # Punto de entrada principal
├── requirements.txt        # Dependencias del proyecto
└── README.md              # Este archivo
```

## Uso

1. Asegúrate de tener las credenciales de Binance configuradas en el archivo `.env`

2. Ejecuta el script principal:
```bash
python main.py
```

El script realizará las siguientes acciones:
- Descargará datos históricos de precios
- Calculará indicadores técnicos
- Generará señales de trading
- Realizará backtesting de la estrategia
- Entrenará un modelo predictivo
- Mostrará gráficos de los resultados

## Personalización

Puedes modificar los siguientes parámetros en `main.py`:
- Rango de fechas para el análisis
- Par de trading (por defecto BTCUSDT)
- Intervalo de tiempo (por defecto 1h)
- Capital inicial
- Tamaño de posición
- Stop loss y take profit

## Contribuir

Las contribuciones son bienvenidas. Por favor, abre un issue para discutir los cambios propuestos.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles. 