from typing import Optional
from datetime import datetime
import pandas as pd
from pathlib import Path

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

def update_price_data(
    existing_filepath: str,
    new_data: pd.DataFrame,
    output_filepath: Optional[str] = None
) -> pd.DataFrame:
    """
    Update existing price data with new data.
    
    Args:
        existing_filepath (str): Path to existing CSV file
        new_data (pd.DataFrame): New price data to be added
        output_filepath (Optional[str]): Path to save updated data. If None, overwrites existing file
        
    Returns:
        pd.DataFrame: Updated price data
    """
    # Load existing data
    existing_df = load_existing_data(existing_filepath)
    
    if existing_df is None:
        # If no existing data, just save the new data
        if output_filepath is None:
            output_filepath = existing_filepath
        new_data.to_csv(output_filepath, index=False)
        return new_data
    
    # Merge and deduplicate data
    updated_df = merge_and_deduplicate(existing_df, new_data)
    
    # Save updated data
    if output_filepath is None:
        output_filepath = existing_filepath
    updated_df.to_csv(output_filepath, index=False)
    
    return updated_df 