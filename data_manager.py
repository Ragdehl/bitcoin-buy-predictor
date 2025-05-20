from typing import Optional, Tuple
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import json

from data.fetch_prices import fetch_price_data
from features.feature_engineering_config import calculate_features

class DataManager:
    """
    Manages Bitcoin price data and technical indicators.
    Handles both initial data download and incremental updates.
    """
    
    def __init__(
        self,
        data_dir: str = 'data',
        data_file: str = 'historical.csv',
        config_file: str = 'features/features_config.json',
        keep_only_main_columns: bool = False
    ):
        """
        Initialize the DataManager.
        
        Args:
            data_dir (str, optional): Directory to store data files. Defaults to 'data'
            data_file (str, optional): Name of the data file containing prices and indicators.
                Defaults to 'historical.csv'
            config_file (str, optional): Path to feature configuration file.
                Defaults to 'features/features_config.json'
            keep_only_main_columns (bool, optional): If True, only keeps OHLCV + timestamp columns.
                Defaults to False
        """
        # Create data directory if it doesn't exist
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Set file paths
        self.data_file = self.data_dir / data_file
        self.config_file = Path(config_file)
        
        # Set main columns flag
        self.keep_only_main_columns = keep_only_main_columns
        
        # Define main columns
        self.main_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Load API credentials
        load_dotenv()
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Binance API credentials not found in .env file")
    
    def _filter_main_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter DataFrame to keep only main columns if keep_only_main_columns is True.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: Filtered DataFrame with only main columns if keep_only_main_columns is True
        """
        if self.keep_only_main_columns:
            return df[self.main_columns]
        return df
    
    def load_existing_data(self) -> Optional[pd.DataFrame]:
        """
        Load existing data if available.
        
        Returns:
            Optional[pd.DataFrame]: DataFrame with price data and indicators, or None if no data exists
        """
        if self.data_file.exists():
            df = pd.read_csv(self.data_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # If keep_only_main_columns is True, filter and save only main columns
            if self.keep_only_main_columns:
                df = df[self.main_columns]
                df.to_csv(self.data_file, index=False)
                print("Filtered historical.csv to contain only main columns")
            
            return df
        return None
    
    def initialize_data(self) -> pd.DataFrame:
        """
        Download initial historical data and calculate indicators.
        Fetches maximum available historical data from Binance.
        
        Returns:
            pd.DataFrame: DataFrame with price data and indicators
        """
        print("Downloading maximum available historical data...")
        
        # Download price data
        price_data, _ = fetch_price_data(
            api_key=self.api_key,
            api_secret=self.api_secret,
            existing_file=str(self.data_file)
        )
        
        if not self.keep_only_main_columns:
            # Clear all column parameters in config file
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            for feature_name in config:
                if 'columns' in config[feature_name]:
                    config[feature_name]['columns'] = []
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
            print("Cleared all column parameters in configuration file")
            
            # Calculate indicators
            print("Calculating technical indicators...")
            try:
                data_with_indicators = calculate_features(
                    df=price_data,
                    config_path=self.config_file
                )
                
                # Verify that indicators were calculated
                indicator_columns = [col for col in data_with_indicators.columns if col not in self.main_columns]
                if not indicator_columns:
                    raise ValueError("No indicator columns were created")
                
                # Save data
                data_with_indicators.to_csv(self.data_file, index=False)
                print("\nInitial data download and processing complete!")
                return data_with_indicators
            except Exception as e:
                print(f"Error calculating indicators: {str(e)}")
                print("Falling back to main columns only")
                self.keep_only_main_columns = True
                return price_data[self.main_columns]
        else:
            # Save only main columns
            price_data = price_data[self.main_columns]
            price_data.to_csv(self.data_file, index=False)
            print("\nInitial data download complete (main columns only)!")
            return price_data
    
    def update_data(self) -> pd.DataFrame:
        """
        Update data with new price information and indicators.
        
        Returns:
            pd.DataFrame: Updated DataFrame with price data and indicators
        """
        print("Checking for new data...")
        
        # Load existing data
        existing_data = self.load_existing_data()
        
        # Fetch new price data
        updated_price_data, new_price_data = fetch_price_data(
            api_key=self.api_key,
            api_secret=self.api_secret,
            existing_file=str(self.data_file)
        )
        
        if new_price_data.empty:
            print("No new data available.")
        
        print(f"Downloaded {len(new_price_data)} new data points.")
        
        if not self.keep_only_main_columns:
            # Clear all column parameters in config file
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            for feature_name in config:
                if 'columns' in config[feature_name]:
                    config[feature_name]['columns'] = []
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
            print("Cleared all column parameters in configuration file")
            
            # Calculate indicators for new data
            print("Calculating technical indicators for new data...")
            try:
                updated_data = calculate_features(
                    df=updated_price_data,
                    config_path=self.config_file,
                    existing_columns=existing_data.columns if existing_data is not None else None
                )
                
                # Verify that indicators were calculated
                indicator_columns = [col for col in updated_data.columns if col not in self.main_columns]
                if not indicator_columns:
                    raise ValueError("No indicator columns were created")
                
            except Exception as e:
                print(f"Error calculating indicators: {str(e)}")
                print("Falling back to main columns only")
                self.keep_only_main_columns = True
                updated_data = updated_price_data[self.main_columns]
        else:
            # Keep only main columns
            updated_data = updated_price_data[self.main_columns]
        
        # Save updated data
        updated_data.to_csv(self.data_file, index=False)
        
        print("\nData update complete!")
        return updated_data

def main():
    """
    Main function to demonstrate usage of DataManager.
    """
    # Initialize data manager
    manager = DataManager(keep_only_main_columns=False)
    
    # Check if we need to initialize data
    data = manager.load_existing_data()
    
    if data is None:
        # Download initial data
        data = manager.initialize_data()
    else:
        # Update existing data
        data = manager.update_data()
    
    # Print some statistics
    print("\nData Statistics:")
    print(f"Total data points: {len(data)}")
    print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(f"Latest price: ${data['close'].iloc[-1]:.2f}")
    if not manager.keep_only_main_columns:
        print(f"Number of technical indicators: {len(data.columns) - 6}")  # 6 is the number of OHLCV + timestamp columns
    else:
        print("Running in main columns only mode")

if __name__ == "__main__":
    main() 