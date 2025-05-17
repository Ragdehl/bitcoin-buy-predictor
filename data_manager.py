from typing import Optional, Tuple
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

from data.fetch_prices import fetch_price_data
from features.feature_engineering import calculate_indicators_incremental

class DataManager:
    """
    Manages Bitcoin price data and technical indicators.
    Handles both initial data download and incremental updates.
    """
    
    def __init__(
        self,
        data_dir: str = 'data',
        price_file: str = 'historical.csv',
        indicators_file: str = 'historical_with_indicators.csv',
        rsi_window: int = 14,
        macd_params: Optional[dict] = None,
        ema_windows: Optional[list] = None,
        bb_params: Optional[dict] = None
    ):
        """
        Initialize the DataManager.
        
        Args:
            data_dir (str, optional): Directory to store data files. Defaults to 'data'
            price_file (str, optional): Name of the price data file. Defaults to 'historical.csv'
            indicators_file (str, optional): Name of the file with indicators. 
                Defaults to 'historical_with_indicators.csv'
            rsi_window (int, optional): RSI window period. Defaults to 14
            macd_params (Optional[dict], optional): MACD parameters. Defaults to None
            ema_windows (Optional[list], optional): List of EMA windows. Defaults to None
            bb_params (Optional[dict], optional): Bollinger Bands parameters. Defaults to None
        """
        # Create data directory if it doesn't exist
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Set file paths
        self.price_file = self.data_dir / price_file
        self.indicators_file = self.data_dir / indicators_file
        
        # Set indicator parameters
        self.rsi_window = rsi_window
        self.macd_params = macd_params or {
            'window_slow': 26,
            'window_fast': 12,
            'window_sign': 9
        }
        self.ema_windows = ema_windows or [20, 50, 200]
        self.bb_params = bb_params or {
            'window': 20,
            'window_dev': 2
        }
        
        # Load API credentials
        load_dotenv()
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Binance API credentials not found in .env file")
    
    def load_existing_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Load existing price data and indicators if available.
        
        Returns:
            Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]: Tuple containing:
                - Price data DataFrame or None
                - Indicators DataFrame or None
        """
        price_data = None
        indicators_data = None
        
        if self.price_file.exists():
            price_data = pd.read_csv(self.price_file)
            price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
        
        if self.indicators_file.exists():
            indicators_data = pd.read_csv(self.indicators_file)
            indicators_data['timestamp'] = pd.to_datetime(indicators_data['timestamp'])
        
        return price_data, indicators_data
    
    def initialize_data(self, days_back: int = 365) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Download initial historical data and calculate indicators.
        
        Args:
            days_back (int, optional): Number of days of historical data to download. 
                Defaults to 365
                
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing:
                - Price data DataFrame
                - Indicators DataFrame
        """
        print(f"Downloading {days_back} days of historical data...")
        
        # Download price data
        price_data, _ = fetch_price_data(
            api_key=self.api_key,
            api_secret=self.api_secret,
            days_back=days_back,
            existing_file=str(self.price_file)
        )
        
        # Calculate indicators
        print("Calculating technical indicators...")
        _, indicators_data = calculate_indicators_incremental(
            new_data=price_data,
            rsi_window=self.rsi_window,
            macd_params=self.macd_params,
            ema_windows=self.ema_windows,
            bb_params=self.bb_params
        )
        
        # Save data
        price_data.to_csv(self.price_file, index=False)
        indicators_data.to_csv(self.indicators_file, index=False)
        
        print("Initial data download and processing complete!")
        return price_data, indicators_data
    
    def update_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Update price data and indicators with new data.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing:
                - Updated price data DataFrame
                - Updated indicators DataFrame
        """
        print("Checking for new data...")
        
        # Load existing data
        price_data, indicators_data = self.load_existing_data()
        
        # Fetch new price data
        updated_price_data, new_price_data = fetch_price_data(
            api_key=self.api_key,
            api_secret=self.api_secret,
            existing_file=str(self.price_file)
        )
        
        if new_price_data.empty:
            print("No new data available.")
            return updated_price_data, indicators_data
        
        print(f"Downloaded {len(new_price_data)} new data points.")
        
        # Calculate indicators for new data
        print("Calculating technical indicators for new data...")
        new_indicators, updated_indicators = calculate_indicators_incremental(
            new_data=new_price_data,
            existing_data=indicators_data,
            rsi_window=self.rsi_window,
            macd_params=self.macd_params,
            ema_windows=self.ema_windows,
            bb_params=self.bb_params
        )
        
        # Save updated data
        updated_indicators.to_csv(self.indicators_file, index=False)
        
        print("Data update complete!")
        return updated_price_data, updated_indicators

def main():
    """
    Main function to demonstrate usage of DataManager.
    """
    # Initialize data manager
    manager = DataManager()
    
    # Check if we need to initialize data
    price_data, indicators_data = manager.load_existing_data()
    
    if price_data is None:
        # Download initial data
        price_data, indicators_data = manager.initialize_data(days_back=365)
    else:
        # Update existing data
        price_data, indicators_data = manager.update_data()
    
    # Print some statistics
    print("\nData Statistics:")
    print(f"Total data points: {len(price_data)}")
    print(f"Date range: {price_data['timestamp'].min()} to {price_data['timestamp'].max()}")
    print(f"Latest price: ${price_data['close'].iloc[-1]:.2f}")

if __name__ == "__main__":
    main() 