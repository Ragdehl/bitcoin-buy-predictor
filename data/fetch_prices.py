from typing import Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
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
    days_back: Optional[int] = None,
    existing_file: str = 'data/historical.csv'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch Bitcoin price data, either initial batch or incremental update.
    
    Args:
        api_key (str): Binance API key
        api_secret (str): Binance API secret
        symbol (str, optional): Trading pair symbol. Defaults to 'BTCUSDT'
        interval (str, optional): Kline interval. Defaults to '1h'
        days_back (Optional[int], optional): Number of days of historical data to download.
            If None, only fetches new data since last available timestamp.
            Defaults to None
        existing_file (str, optional): Path to existing data file. 
            Defaults to 'data/historical.csv'
            
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing:
            - Updated DataFrame with all data
            - DataFrame with only new data
            
    Raises:
        Exception: If there's an error fetching or processing data
    """
    # Initialize client
    client = initialize_binance_client(api_key, api_secret)
    
    # Calculate date range
    end_date = datetime.now()
    
    if days_back is not None:
        # Fetch initial batch
        start_date = end_date - timedelta(days=days_back)
        new_df = fetch_historical_klines(client, symbol, start_date, end_date, interval)
        updated_df = new_df
    else:
        # Fetch incremental update
        existing_df = load_existing_data(existing_file)
        
        if existing_df is not None:
            start_date = get_latest_timestamp(existing_df)
        else:
            start_date = end_date - timedelta(hours=24)  # Default to 24h if no existing data
            
        new_df = fetch_historical_klines(client, symbol, start_date, end_date, interval)
        
        if existing_df is not None:
            updated_df = merge_and_deduplicate(existing_df, new_df)
        else:
            updated_df = new_df
    
    # Save updated data
    updated_df.to_csv(existing_file, index=False)
    
    return updated_df, new_df 