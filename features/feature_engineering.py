from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import ta

def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        df (pd.DataFrame): DataFrame with 'close' prices
        window (int, optional): RSI window period. Defaults to 14
        
    Returns:
        pd.Series: RSI values
    """
    return ta.momentum.RSIIndicator(
        close=df['close'],
        window=window
    ).rsi()

def calculate_macd(
    df: pd.DataFrame,
    window_slow: int = 26,
    window_fast: int = 12,
    window_sign: int = 9
) -> Dict[str, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        df (pd.DataFrame): DataFrame with 'close' prices
        window_slow (int, optional): Slow period. Defaults to 26
        window_fast (int, optional): Fast period. Defaults to 12
        window_sign (int, optional): Signal period. Defaults to 9
        
    Returns:
        Dict[str, pd.Series]: Dictionary containing:
            - macd: MACD line
            - signal: Signal line
            - histogram: MACD histogram
    """
    macd = ta.trend.MACD(
        close=df['close'],
        window_slow=window_slow,
        window_fast=window_fast,
        window_sign=window_sign
    )
    
    return {
        'macd': macd.macd(),
        'signal': macd.macd_signal(),
        'histogram': macd.macd_diff()
    }

def calculate_ema(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        df (pd.DataFrame): DataFrame with 'close' prices
        window (int): EMA window period
        
    Returns:
        pd.Series: EMA values
    """
    return ta.trend.EMAIndicator(
        close=df['close'],
        window=window
    ).ema_indicator()

def calculate_bollinger_bands(
    df: pd.DataFrame,
    window: int = 20,
    window_dev: int = 2
) -> Dict[str, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        df (pd.DataFrame): DataFrame with 'close' prices
        window (int, optional): Moving average window. Defaults to 20
        window_dev (int, optional): Number of standard deviations. Defaults to 2
        
    Returns:
        Dict[str, pd.Series]: Dictionary containing:
            - upper: Upper band
            - middle: Middle band (SMA)
            - lower: Lower band
    """
    bollinger = ta.volatility.BollingerBands(
        close=df['close'],
        window=window,
        window_dev=window_dev
    )
    
    return {
        'upper': bollinger.bollinger_hband(),
        'middle': bollinger.bollinger_mavg(),
        'lower': bollinger.bollinger_lband()
    }

def add_technical_indicators(
    df: pd.DataFrame,
    rsi_window: int = 14,
    macd_params: Optional[Dict[str, int]] = None,
    ema_windows: Optional[List[int]] = None,
    bb_params: Optional[Dict[str, int]] = None
) -> pd.DataFrame:
    """
    Add technical indicators to the price DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        rsi_window (int, optional): RSI window period. Defaults to 14
        macd_params (Optional[Dict[str, int]], optional): MACD parameters. Defaults to None
        ema_windows (Optional[List[int]], optional): List of EMA windows. Defaults to None
        bb_params (Optional[Dict[str, int]], optional): Bollinger Bands parameters. Defaults to None
        
    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Calculate RSI
    df['rsi'] = calculate_rsi(df, window=rsi_window)
    
    # Calculate MACD if parameters provided
    if macd_params is not None:
        macd_data = calculate_macd(
            df,
            window_slow=macd_params.get('window_slow', 26),
            window_fast=macd_params.get('window_fast', 12),
            window_sign=macd_params.get('window_sign', 9)
        )
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_histogram'] = macd_data['histogram']
    
    # Calculate EMAs if windows provided
    if ema_windows is not None:
        for window in ema_windows:
            df[f'ema_{window}'] = calculate_ema(df, window=window)
    
    # Calculate Bollinger Bands if parameters provided
    if bb_params is not None:
        bb_data = calculate_bollinger_bands(
            df,
            window=bb_params.get('window', 20),
            window_dev=bb_params.get('window_dev', 2)
        )
        df['bb_upper'] = bb_data['upper']
        df['bb_middle'] = bb_data['middle']
        df['bb_lower'] = bb_data['lower']
    
    return df 