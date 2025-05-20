from typing import Dict, Any, Optional, List, Union
import json
from pathlib import Path
import pandas as pd
import ta
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class IndicatorConfig:
    """
    Configuration for a technical indicator.
    
    Attributes:
        type (str): Type of indicator (RSI, MACD, EMA, etc.)
        params (Dict[str, Any]): Parameters for the indicator
    """
    type: str
    params: Dict[str, Any]

class IndicatorCalculator(ABC):
    """
    Abstract base class for indicator calculators.
    """
    
    @abstractmethod
    def validate_params(self, params: Dict[str, Any]) -> None:
        """
        Validate that all required parameters are present.
        
        Args:
            params (Dict[str, Any]): Parameters to validate
            
        Raises:
            ValueError: If required parameters are missing
        """
        pass
    
    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
        """
        Calculate the indicator values.
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            
        Returns:
            Union[pd.Series, Dict[str, pd.Series]]: Calculated indicator values
        """
        pass

class RSICalculator(IndicatorCalculator):
    """
    Calculator for Relative Strength Index (RSI).
    
    Technical Details:
    RSI is a momentum oscillator that measures the speed and change of price movements.
    It oscillates between 0 and 100, with values above 70 indicating overbought conditions
    and values below 30 indicating oversold conditions.
    
    Formula:
    1. Calculate price changes: delta = close - close.shift(1)
    2. Separate gains and losses: gains = delta.where(delta > 0, 0)
    3. Calculate average gains and losses over the window period
    4. RSI = 100 - (100 / (1 + (avg_gains / avg_losses)))
    
    The indicator uses exponential moving averages for smoothing.
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize RSI calculator.
        
        Args:
            params (Dict[str, Any]): Parameters for RSI calculation
        """
        self.validate_params(params)
        self.window = params['window']
        self.enabled = params.get('enabled', True)
    
    def validate_params(self, params: Dict[str, Any]) -> None:
        """
        Validate RSI parameters.
        
        Args:
            params (Dict[str, Any]): Parameters to validate
            
        Raises:
            ValueError: If 'window' parameter is missing
        """
        if 'window' not in params:
            raise ValueError("RSI indicator requires 'window' parameter")
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate RSI values.
        
        Args:
            df (pd.DataFrame): DataFrame with 'close' prices
            
        Returns:
            pd.Series: RSI values
        """
        if not self.enabled:
            return pd.Series(index=df.index)
            
        return ta.momentum.RSIIndicator(
            close=df['close'],
            window=self.window
        ).rsi()

class MACDCalculator(IndicatorCalculator):
    """
    Calculator for Moving Average Convergence Divergence (MACD).
    
    Technical Details:
    MACD is a trend-following momentum indicator that shows the relationship between
    two moving averages of an asset's price.
    
    Components:
    1. MACD Line: Difference between fast and slow EMAs
       MACD = EMA(fast_period) - EMA(slow_period)
    2. Signal Line: EMA of the MACD line
       Signal = EMA(MACD, signal_period)
    3. Histogram: Difference between MACD and Signal lines
       Histogram = MACD - Signal
    
    Trading signals:
    - Bullish: MACD crosses above Signal
    - Bearish: MACD crosses below Signal
    - Divergence: Price and MACD move in opposite directions
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize MACD calculator.
        
        Args:
            params (Dict[str, Any]): Parameters for MACD calculation
        """
        self.validate_params(params)
        self.fast = params['fast']
        self.slow = params['slow']
        self.signal = params['signal']
        self.enabled = params.get('enabled', True)
    
    def validate_params(self, params: Dict[str, Any]) -> None:
        """
        Validate MACD parameters.
        
        Args:
            params (Dict[str, Any]): Parameters to validate
            
        Raises:
            ValueError: If any of 'fast', 'slow', or 'signal' parameters are missing
        """
        required_params = ['fast', 'slow', 'signal']
        missing_params = [param for param in required_params if param not in params]
        if missing_params:
            raise ValueError(f"MACD indicator requires parameters: {', '.join(missing_params)}")
    
    def calculate(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate MACD values.
        
        Args:
            df (pd.DataFrame): DataFrame with 'close' prices
            
        Returns:
            Dict[str, pd.Series]: Dictionary containing:
                - macd: MACD line
                - signal: Signal line
                - histogram: MACD histogram
        """
        if not self.enabled:
            return {
                'macd': pd.Series(index=df.index),
                'signal': pd.Series(index=df.index),
                'histogram': pd.Series(index=df.index)
            }
            
        macd = ta.trend.MACD(
            close=df['close'],
            window_slow=self.slow,
            window_fast=self.fast,
            window_sign=self.signal
        )
        
        return {
            'macd': macd.macd(),
            'signal': macd.macd_signal(),
            'histogram': macd.macd_diff()
        }

class EMACalculator(IndicatorCalculator):
    """
    Calculator for Exponential Moving Average (EMA).
    
    Technical Details:
    EMA is a type of moving average that gives more weight to recent prices,
    making it more responsive to price changes than Simple Moving Average (SMA).
    
    Formula:
    1. Multiplier = (2 / (window + 1))
    2. EMA = (current_price - previous_EMA) * multiplier + previous_EMA
    
    The first EMA value is typically calculated using SMA.
    EMA responds faster to price changes than SMA because it gives more
    weight to recent prices, with the weight decreasing exponentially
    for older prices.
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize EMA calculator.
        
        Args:
            params (Dict[str, Any]): Parameters for EMA calculation
        """
        self.validate_params(params)
        self.window = params['window']
        self.enabled = params.get('enabled', True)
    
    def validate_params(self, params: Dict[str, Any]) -> None:
        """
        Validate EMA parameters.
        
        Args:
            params (Dict[str, Any]): Parameters to validate
            
        Raises:
            ValueError: If 'window' parameter is missing
        """
        if 'window' not in params:
            raise ValueError("EMA indicator requires 'window' parameter")
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate EMA values.
        
        Args:
            df (pd.DataFrame): DataFrame with 'close' prices
            
        Returns:
            pd.Series: EMA values
        """
        if not self.enabled:
            return pd.Series(index=df.index)
            
        return ta.trend.EMAIndicator(
            close=df['close'],
            window=self.window
        ).ema_indicator()

class BollingerBandsCalculator(IndicatorCalculator):
    """
    Calculator for Bollinger Bands.
    
    Technical Details:
    Bollinger Bands consist of three lines:
    1. Middle Band: N-period Simple Moving Average (SMA)
    2. Upper Band: Middle Band + (K * N-period standard deviation)
    3. Lower Band: Middle Band - (K * N-period standard deviation)
    
    Where:
    - N is the window period (typically 20)
    - K is the number of standard deviations (typically 2)
    
    Trading signals:
    - Price touching upper band: Potential overbought condition
    - Price touching lower band: Potential oversold condition
    - Band width (distance between upper and lower bands):
      - Narrowing: Decreasing volatility
      - Widening: Increasing volatility
    
    The bands adapt to market conditions by expanding during volatile
    periods and contracting during less volatile periods.
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize Bollinger Bands calculator.
        
        Args:
            params (Dict[str, Any]): Parameters for Bollinger Bands calculation
        """
        self.validate_params(params)
        self.window = params['window']
        self.std_dev = params['std_dev']
        self.enabled = params.get('enabled', True)
    
    def validate_params(self, params: Dict[str, Any]) -> None:
        """
        Validate Bollinger Bands parameters.
        
        Args:
            params (Dict[str, Any]): Parameters to validate
            
        Raises:
            ValueError: If 'window' or 'std_dev' parameters are missing
        """
        required_params = ['window', 'std_dev']
        missing_params = [param for param in required_params if param not in params]
        if missing_params:
            raise ValueError(f"Bollinger Bands indicator requires parameters: {', '.join(missing_params)}")
    
    def calculate(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands values.
        
        Args:
            df (pd.DataFrame): DataFrame with 'close' prices
            
        Returns:
            Dict[str, pd.Series]: Dictionary containing:
                - upper: Upper band
                - middle: Middle band (SMA)
                - lower: Lower band
        """
        if not self.enabled:
            return {
                'upper': pd.Series(index=df.index),
                'middle': pd.Series(index=df.index),
                'lower': pd.Series(index=df.index)
            }
            
        bollinger = ta.volatility.BollingerBands(
            close=df['close'],
            window=self.window,
            window_dev=self.std_dev
        )
        
        return {
            'upper': bollinger.bollinger_hband(),
            'middle': bollinger.bollinger_mavg(),
            'lower': bollinger.bollinger_lband()
        }

class CandleShapeCalculator(IndicatorCalculator):
    """
    Calculator for individual candle shape features.
    
    Technical Details:
    Calculates various metrics about individual candle shapes:
    - Body size and position
    - Wick lengths
    - Color and ratios
    
    Features:
    - body: Absolute size of the candle body
    - upper_wick: Length of the upper wick
    - lower_wick: Length of the lower wick
    - range: Total range of the candle
    - color: 1 for green (bullish), 0 for red (bearish)
    - body_ratio: Proportion of body to total range
    - upper_ratio: Proportion of upper wick to total range
    - lower_ratio: Proportion of lower wick to total range
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize CandleShape calculator.
        
        Args:
            params (Dict[str, Any]): Parameters including 'include_columns' list
        """
        self.validate_params(params)
        self.include_columns = params['include_columns']
        self.enabled = params.get('enabled', True)
    
    def validate_params(self, params: Dict[str, Any]) -> None:
        """
        Validate CandleShape parameters.
        
        Args:
            params (Dict[str, Any]): Parameters to validate
            
        Raises:
            ValueError: If 'include_columns' is missing or invalid
        """
        if 'include_columns' not in params:
            raise ValueError("CandleShape indicator requires 'include_columns' parameter")
        
        valid_columns = ["body", "upper_wick", "lower_wick", "range", "color", 
                        "body_ratio", "upper_ratio", "lower_ratio"]
        invalid_columns = [col for col in params['include_columns'] if col not in valid_columns]
        if invalid_columns:
            raise ValueError(f"Invalid columns for CandleShape: {invalid_columns}")
    
    def calculate(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate candle shape features.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data
            
        Returns:
            Dict[str, pd.Series]: Dictionary of calculated features
        """
        if not self.enabled:
            return {col: pd.Series(index=df.index) for col in self.include_columns}
            
        # Calculate basic candle metrics
        body = abs(df['close'] - df['open'])
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        candle_range = df['high'] - df['low']
        color = (df['close'] > df['open']).astype(int)
        
        # Calculate ratios (avoid division by zero)
        body_ratio = body / candle_range.replace(0, float('inf'))
        upper_ratio = upper_wick / candle_range.replace(0, float('inf'))
        lower_ratio = lower_wick / candle_range.replace(0, float('inf'))
        
        # Create feature dictionary
        features = {
            'body': body,
            'upper_wick': upper_wick,
            'lower_wick': lower_wick,
            'range': candle_range,
            'color': color,
            'body_ratio': body_ratio,
            'upper_ratio': upper_ratio,
            'lower_ratio': lower_ratio
        }
        
        # Return only requested features
        return {k: v for k, v in features.items() if k in self.include_columns}

class CandlePatternsCalculator(IndicatorCalculator):
    """
    Calculator for multi-candle pattern features.
    
    Technical Details:
    Identifies common candlestick patterns:
    - Engulfing patterns (bullish/bearish)
    - Three white soldiers (bullish)
    - Three black crows (bearish)
    - Doji followed by bullish candle
    
    Each pattern is represented as a boolean flag (0/1).
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize CandlePatterns calculator.
        
        Args:
            params (Dict[str, Any]): Parameters including 'include_patterns' list and 'doji_threshold'
        """
        self.validate_params(params)
        self.include_patterns = params['include_patterns']
        self.doji_threshold = params['doji_threshold']
        self.enabled = params.get('enabled', True)
    
    def validate_params(self, params: Dict[str, Any]) -> None:
        """
        Validate CandlePatterns parameters.
        
        Args:
            params (Dict[str, Any]): Parameters to validate
            
        Raises:
            ValueError: If required parameters are missing or invalid
        """
        if 'include_patterns' not in params:
            raise ValueError("CandlePatterns indicator requires 'include_patterns' parameter")
        
        valid_patterns = ["bullish_engulfing", "bearish_engulfing", 
                         "three_white_soldiers", "three_black_crows", 
                         "doji_then_bull"]
        invalid_patterns = [p for p in params['include_patterns'] if p not in valid_patterns]
        if invalid_patterns:
            raise ValueError(f"Invalid patterns for CandlePatterns: {invalid_patterns}")
        
        if 'doji_threshold' not in params:
            raise ValueError("CandlePatterns indicator requires 'doji_threshold' parameter")
    
    def calculate(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate candlestick pattern features.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data
            
        Returns:
            Dict[str, pd.Series]: Dictionary of pattern flags
        """
        if not self.enabled:
            return {pattern: pd.Series(index=df.index) for pattern in self.include_patterns}
            
        features = {}
        
        if "bullish_engulfing" in self.include_patterns:
            # Current candle is green and fully engulfs previous red candle
            current_green = df['close'] > df['open']
            prev_red = df['open'].shift(1) > df['close'].shift(1)
            current_higher = df['close'] > df['open'].shift(1)
            current_lower = df['open'] < df['close'].shift(1)
            features['bullish_engulfing'] = (current_green & prev_red & current_higher & current_lower).astype(int)
        
        if "bearish_engulfing" in self.include_patterns:
            # Current candle is red and fully engulfs previous green candle
            current_red = df['close'] < df['open']
            prev_green = df['open'].shift(1) < df['close'].shift(1)
            current_higher = df['open'] > df['close'].shift(1)
            current_lower = df['close'] < df['open'].shift(1)
            features['bearish_engulfing'] = (current_red & prev_green & current_higher & current_lower).astype(int)
        
        if "three_white_soldiers" in self.include_patterns:
            # Last 3 candles are all green, each closing higher than the previous
            green_candles = df['close'] > df['open']
            higher_closes = df['close'] > df['close'].shift(1)
            features['three_white_soldiers'] = (
                green_candles & 
                green_candles.shift(1) & 
                green_candles.shift(2) &
                higher_closes &
                higher_closes.shift(1)
            ).astype(int)
        
        if "three_black_crows" in self.include_patterns:
            # Last 3 candles are all red, each closing lower than the previous
            red_candles = df['close'] < df['open']
            lower_closes = df['close'] < df['close'].shift(1)
            features['three_black_crows'] = (
                red_candles & 
                red_candles.shift(1) & 
                red_candles.shift(2) &
                lower_closes &
                lower_closes.shift(1)
            ).astype(int)
        
        if "doji_then_bull" in self.include_patterns:
            # Small body candle followed by strong green candle
            body_size = abs(df['close'] - df['open'])
            candle_range = df['high'] - df['low']
            doji = (body_size / candle_range.replace(0, float('inf'))) < self.doji_threshold
            strong_bull = (df['close'] > df['open']) & (body_size > body_size.mean())
            features['doji_then_bull'] = (doji.shift(1) & strong_bull).astype(int)
        
        return features

def get_calculator(config: Dict[str, Any]) -> IndicatorCalculator:
    """
    Get the appropriate calculator for an indicator configuration.
    
    Args:
        config (Dict[str, Any]): Indicator configuration dictionary
        
    Returns:
        IndicatorCalculator: Calculator instance
        
    Raises:
        ValueError: If indicator type is not supported or if required parameters are missing
    """
    if 'type' not in config:
        raise ValueError("Indicator configuration must include 'type' field")
    
    # Create a copy of the config to avoid modifying the original
    config_copy = config.copy()
    indicator_type = config_copy.pop('type')  # Remove type from config to get remaining params
    
    print(f"Creating calculator for {indicator_type} with params: {config_copy}")
    
    if indicator_type == "RSI":
        return RSICalculator(params=config_copy)
    elif indicator_type == "MACD":
        return MACDCalculator(params=config_copy)
    elif indicator_type == "EMA":
        return EMACalculator(params=config_copy)
    elif indicator_type == "BollingerBands":
        return BollingerBandsCalculator(params=config_copy)
    elif indicator_type == "CandleShape":
        return CandleShapeCalculator(params=config_copy)
    elif indicator_type == "CandlePatterns":
        return CandlePatternsCalculator(params=config_copy)
    else:
        raise ValueError(f"Unsupported indicator type: {indicator_type}")

def load_config(config_path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """
    Load indicator configurations from JSON file.
    
    Args:
        config_path (Union[str, Path]): Path to configuration file
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping column names to configurations
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        json.JSONDecodeError: If configuration file is not valid JSON
    """
    with open(config_path, 'r') as f:
        return json.load(f)

def calculate_features(
    df: pd.DataFrame,
    config_path: Union[str, Path],
    existing_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate technical indicators based on configuration.
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        config_path (Union[str, Path]): Path to configuration file
        existing_columns (Optional[List[str]], optional): List of existing indicator columns.
            If provided, only missing indicators will be calculated. Defaults to None
            
    Returns:
        pd.DataFrame: DataFrame with added technical indicators
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        json.JSONDecodeError: If configuration file is not valid JSON
        ValueError: If indicator type is not supported or if required parameters are missing
    """
    # Load configurations
    configs = load_config(config_path)
    
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Get list of existing columns if not provided
    if existing_columns is None:
        existing_columns = df.columns.tolist()
    
    # First, drop columns for disabled features
    for column_name, config in configs.items():
        if not config.get('enabled', True) and 'columns' in config:
            # Drop all columns associated with this feature
            columns_to_drop = [col for col in config['columns'] if col in df.columns]
            if columns_to_drop:
                df = df.drop(columns=columns_to_drop)
                # Update existing_columns list
                existing_columns = [col for col in existing_columns if col not in columns_to_drop]
                # Remove columns from config
                configs[column_name]['columns'] = []
                print(f"Removed columns from {column_name} configuration: {columns_to_drop}")
    
    # Save updated configuration after dropping columns
    with open(config_path, 'w') as f:
        json.dump(configs, f, indent=4)
    
    # Calculate each indicator
    for column_name, config in configs.items():
        # Skip if feature is disabled
        if not config.get('enabled', True):
            continue
            
        # Skip if column already exists
        if column_name in existing_columns:
            continue
        
        try:
            # Get calculator and calculate values
            calculator = get_calculator(config)
            values = calculator.calculate(df)
            
            # Track created columns for this feature
            created_columns = []
            
            # Add values to DataFrame
            if isinstance(values, pd.Series):
                df[column_name] = values
                created_columns.append(column_name)
                print(f"Added column: {column_name}")
            else:
                # For multi-column indicators (like MACD and Bollinger Bands)
                for suffix, series in values.items():
                    column_name_with_suffix = f"{column_name}_{suffix}"
                    df[column_name_with_suffix] = series
                    created_columns.append(column_name_with_suffix)
                    print(f"Added column: {column_name_with_suffix}")
            
            # Update the config file with the actual columns created
            if created_columns:
                configs[column_name]['columns'] = sorted(created_columns)
                # Save updated configuration
                with open(config_path, 'w') as f:
                    json.dump(configs, f, indent=4)
                print(f"Updated configuration for {column_name} with columns: {created_columns}")
                
        except Exception as e:
            print(f"Error calculating {column_name}: {str(e)}")
            continue
    
    # Verify that we have new columns
    new_columns = [col for col in df.columns if col not in existing_columns]
    if not new_columns:
        print("Warning: No new columns were created!")
        print("Existing columns:", existing_columns)
        print("Current columns:", df.columns.tolist())
    else:
        print(f"Successfully created {len(new_columns)} new columns:")
        for col in new_columns:
            print(f"  - {col}")
    
    return df 