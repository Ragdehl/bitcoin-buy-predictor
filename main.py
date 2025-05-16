from typing import Dict, Optional
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

from data.fetch_prices import initialize_binance_client, fetch_historical_klines, save_klines_to_csv
from data.update_data import update_price_data
from features.feature_engineering import add_technical_indicators
from models.backtesting import generate_signals, run_backtest
from models.prediction import prepare_features, train_model, predict_signals, evaluate_predictions
from visualization.plotting import (
    plot_price_and_volume,
    plot_technical_indicators,
    plot_bollinger_bands,
    plot_macd
)

def load_environment() -> Dict[str, str]:
    """
    Load environment variables from .env file.
    
    Returns:
        Dict[str, str]: Dictionary with environment variables
    """
    load_dotenv()
    
    return {
        'api_key': os.getenv('BINANCE_API_KEY'),
        'api_secret': os.getenv('BINANCE_API_SECRET')
    }

def main(
    start_date: datetime,
    end_date: datetime,
    symbol: str = 'BTCUSDT',
    interval: str = '1h',
    data_file: str = 'data/btc_prices.csv',
    initial_capital: float = 10000.0,
    position_size: float = 0.1,
    stop_loss: Optional[float] = 0.02,
    take_profit: Optional[float] = 0.04
) -> None:
    """
    Main function to run the complete workflow.
    
    Args:
        start_date (datetime): Start date for historical data
        end_date (datetime): End date for historical data
        symbol (str, optional): Trading pair symbol. Defaults to 'BTCUSDT'
        interval (str, optional): Kline interval. Defaults to '1h'
        data_file (str, optional): Path to save price data. Defaults to 'data/btc_prices.csv'
        initial_capital (float, optional): Initial capital for backtest. Defaults to 10000.0
        position_size (float, optional): Position size as fraction of capital. Defaults to 0.1
        stop_loss (Optional[float], optional): Stop loss percentage. Defaults to 0.02
        take_profit (Optional[float], optional): Take profit percentage. Defaults to 0.04
    """
    # Load environment variables
    env = load_environment()
    
    # Initialize Binance client
    client = initialize_binance_client(env['api_key'], env['api_secret'])
    
    # Fetch historical data
    df = fetch_historical_klines(client, symbol, start_date, end_date, interval)
    
    # Update existing data
    df = update_price_data(data_file, df)
    
    # Add technical indicators
    df = add_technical_indicators(
        df,
        rsi_window=14,
        macd_params={'window_slow': 26, 'window_fast': 12, 'window_sign': 9},
        ema_windows=[20, 50, 200],
        bb_params={'window': 20, 'window_dev': 2}
    )
    
    # Generate trading signals
    df = generate_signals(
        df,
        rsi_oversold=30,
        rsi_overbought=70,
        macd_signal_threshold=0
    )
    
    # Run backtest
    backtest_result = run_backtest(
        df,
        initial_capital=initial_capital,
        position_size=position_size,
        stop_loss=stop_loss,
        take_profit=take_profit
    )
    
    # Print backtest results
    metrics = backtest_result.calculate_metrics()
    print("\nBacktest Results:")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
    
    # Train prediction model
    X, y = prepare_features(df)
    model, scaler, model_metrics = train_model(X, y)
    
    # Generate predictions
    df = predict_signals(model, scaler, df, df.select_dtypes(include=['float64']).columns.tolist())
    
    # Evaluate predictions
    eval_metrics = evaluate_predictions(df)
    print("\nPrediction Results:")
    print(f"Accuracy: {eval_metrics['accuracy']:.2%}")
    
    # Plot results
    plot_price_and_volume(df)
    plot_technical_indicators(df, ['rsi', 'ema_20', 'ema_50', 'ema_200'])
    plot_bollinger_bands(df)
    plot_macd(df)

if __name__ == "__main__":
    # Example usage
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Last 30 days
    
    main(
        start_date=start_date,
        end_date=end_date,
        symbol='BTCUSDT',
        interval='1h',
        data_file='data/btc_prices.csv',
        initial_capital=10000.0,
        position_size=0.1,
        stop_loss=0.02,
        take_profit=0.04
    ) 