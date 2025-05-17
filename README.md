# Bitcoin Trading Analysis

A Python project for analyzing Bitcoin price data and generating trading signals using technical indicators and machine learning.

## Features

- Download historical Bitcoin price data from Binance
- Calculate technical indicators (RSI, MACD, EMAs, Bollinger Bands)
- Generate trading signals based on technical analysis
- Backtest trading strategies with stop loss and take profit
- Train machine learning models for price prediction
- Visualize results and performance metrics

## Requirements

- Python 3.8+
- Binance account with API credentials
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bitcoin-buy-predictor.git
cd bitcoin-buy-predictor
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your Binance API credentials:
```
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
```

## Project Structure

```
bitcoin-buy-predictor/
├── data/
│   └── fetch_prices.py      # Functions for downloading and updating price data
├── features/
│   └── feature_engineering.py  # Technical indicators and feature calculation
├── models/
│   ├── backtesting.py      # Strategy backtesting implementation
│   └── prediction.py       # Machine learning model for price prediction
├── visualization/
│   └── plotting.py         # Functions for creating charts and visualizations
├── .env                    # Environment variables (API credentials)
├── .gitignore             # Git ignore file
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Usage

### Downloading Price Data

To download historical price data:

```python
from data.fetch_prices import fetch_price_data
import os
from dotenv import load_dotenv

# Load API credentials
load_dotenv()
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

# Download 1 year of historical data
df, _ = fetch_price_data(
    api_key=api_key,
    api_secret=api_secret,
    days_back=365,
    interval='1h'
)

# Update with new data
df, new_data = fetch_price_data(
    api_key=api_key,
    api_secret=api_secret
)
```

### Calculating Technical Indicators

```python
from features.feature_engineering import calculate_indicators

# Calculate technical indicators
df_with_indicators = calculate_indicators(df)
```

### Backtesting Strategies

```python
from models.backtesting import backtest_strategy

# Run backtest
results = backtest_strategy(
    df=df_with_indicators,
    initial_capital=10000,
    position_size=0.1,
    stop_loss=0.02,
    take_profit=0.04
)
```

### Making Predictions

```python
from models.prediction import train_model, make_prediction

# Train model
model = train_model(df_with_indicators)

# Make prediction
prediction = make_prediction(model, latest_data)
```

### Visualizing Results

```python
from visualization.plotting import plot_results

# Create visualization
plot_results(df_with_indicators, results)
```

## Customization

You can customize various parameters in the code:

- Trading pair (default: BTCUSDT)
- Time interval (default: 1h)
- Technical indicator parameters
- Backtesting parameters (initial capital, position size, stop loss, take profit)
- Machine learning model parameters

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 