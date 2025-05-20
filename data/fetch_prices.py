from typing import Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
from binance.client import Client
from binance.exceptions import BinanceAPIException
import os

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
            
    Raises:
        Exception: If there's an error fetching data from Binance
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

def load_existing_data(filepath: str) -> Optional[pd.DataFrame]:
    """
    Load existing price data from CSV file.
    
    Args:
        filepath (str): Path to the CSV file containing price data
        
    Returns:
        Optional[pd.DataFrame]: DataFrame with existing data or None if file doesn't exist
    """
    if not Path(filepath).exists():
        return None
        
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def get_latest_timestamp(df: pd.DataFrame) -> datetime:
    """
    Get the latest timestamp from the existing data.
    
    Args:
        df (pd.DataFrame): DataFrame containing price data
        
    Returns:
        datetime: Latest timestamp in the data
    """
    return df['timestamp'].max()

def merge_and_deduplicate(
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge existing and new data, removing duplicates.
    
    Args:
        existing_df (pd.DataFrame): Existing price data
        new_df (pd.DataFrame): New price data to be added
        
    Returns:
        pd.DataFrame: Combined data with duplicates removed
    """
    # Combine dataframes
    combined_df = pd.concat([existing_df, new_df])
    
    # Remove duplicates based on timestamp
    combined_df = combined_df.drop_duplicates(subset=['timestamp'])
    
    # Sort by timestamp
    combined_df = combined_df.sort_values('timestamp')
    
    return combined_df

def fetch_price_data(
    api_key: str,
    api_secret: str,
    symbol: str = 'BTCUSDT',
    interval: str = '1h',
    existing_file: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch historical price data from Binance.
    If existing_file is provided, only fetches new data since the last entry.
    Otherwise, fetches maximum available historical data.
    
    Args:
        api_key (str): Binance API key
        api_secret (str): Binance API secret
        symbol (str, optional): Trading pair symbol. Defaults to 'BTCUSDT'
        interval (str, optional): Kline interval. Defaults to '1h'
        existing_file (Optional[str], optional): Path to existing data file.
            If provided, only fetches new data. Defaults to None
            
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing:
            - Complete DataFrame with all data
            - DataFrame with only new data (empty if no new data)
    """
    # Initialize Binance client
    client = Client(api_key, api_secret)
    
    # Get the earliest available data point
    earliest_timestamp = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        limit=1,
        start_str="2017-01-01"  # Binance started in 2017
    )[0][0]
    
    # Convert to datetime
    start_date = datetime.fromtimestamp(earliest_timestamp / 1000)
    end_date = datetime.now()
    
    # Load existing data if available
    existing_data = None
    if existing_file and os.path.exists(existing_file):
        existing_data = pd.read_csv(existing_file)
        existing_data['timestamp'] = pd.to_datetime(existing_data['timestamp'])
        start_date = existing_data['timestamp'].max()
    
    # Fetch klines
    klines = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_str=start_date.strftime('%Y-%m-%d %H:%M:%S'),
        end_str=end_date.strftime('%Y-%m-%d %H:%M:%S')
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignored'
    ])
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Convert numeric columns
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
    
    # Drop unnecessary columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # If we have existing data, merge and sort
    if existing_data is not None:
        # Combine existing and new data
        df = pd.concat([existing_data, df])
        # Remove duplicates based on timestamp
        df = df.drop_duplicates(subset=['timestamp'])
        # Sort by timestamp
        df = df.sort_values('timestamp')
        # Get only new data
        new_data = df[df['timestamp'] > start_date]
    else:
        new_data = df
    
    return df, new_data 