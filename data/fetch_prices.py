from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException

def initialize_binance_client(api_key: str, api_secret: str) -> Client:
    """
    Initialize and return a Binance client instance.
    
    Args:
        api_key (str): Binance API key
        api_secret (str): Binance API secret
        
    Returns:
        Client: Initialized Binance client instance
    """
    return Client(api_key, api_secret)

def fetch_historical_klines(
    client: Client,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    interval: str = '1h'
) -> pd.DataFrame:
    """
    Fetch historical klines (candlestick data) from Binance.
    
    Args:
        client (Client): Initialized Binance client
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
        start_date (datetime): Start date for historical data
        end_date (datetime): End date for historical data
        interval (str, optional): Kline interval. Defaults to '1h'
        
    Returns:
        pd.DataFrame: DataFrame containing OHLCV data with columns:
            - timestamp: datetime
            - open: float
            - high: float
            - low: float
            - close: float
            - volume: float
    """
    try:
        klines = client.get_historical_klines(
            symbol,
            interval,
            start_date.strftime("%d %b %Y %H:%M:%S"),
            end_date.strftime("%d %b %Y %H:%M:%S")
        )
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Convert string values to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        # Select and reorder columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        return df
        
    except BinanceAPIException as e:
        raise Exception(f"Error fetching data from Binance: {str(e)}")

def save_klines_to_csv(df: pd.DataFrame, filepath: str) -> None:
    """
    Save klines data to a CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame containing OHLCV data
        filepath (str): Path where the CSV file will be saved
    """
    df.to_csv(filepath, index=False) 