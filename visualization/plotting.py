from typing import List, Optional, Dict
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_price_and_volume(
    df: pd.DataFrame,
    title: str = "BTC Price and Volume",
    figsize: tuple = (12, 8)
) -> None:
    """
    Plot price and volume data.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        title (str, optional): Plot title. Defaults to "BTC Price and Volume"
        figsize (tuple, optional): Figure size. Defaults to (12, 8)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot price
    ax1.plot(df['timestamp'], df['close'], label='Close Price')
    ax1.set_title(title)
    ax1.set_ylabel('Price (USDT)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot volume
    ax2.bar(df['timestamp'], df['volume'], label='Volume')
    ax2.set_ylabel('Volume')
    ax2.grid(True)
    ax2.legend()
    
    # Format x-axis
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_technical_indicators(
    df: pd.DataFrame,
    indicators: List[str],
    title: str = "Technical Indicators",
    figsize: tuple = (12, 8)
) -> None:
    """
    Plot technical indicators.
    
    Args:
        df (pd.DataFrame): DataFrame with price and indicator data
        indicators (List[str]): List of indicator names to plot
        title (str, optional): Plot title. Defaults to "Technical Indicators"
        figsize (tuple, optional): Figure size. Defaults to (12, 8)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot price
    ax.plot(df['timestamp'], df['close'], label='Close Price', alpha=0.5)
    
    # Plot indicators
    for indicator in indicators:
        if indicator in df.columns:
            ax.plot(df['timestamp'], df[indicator], label=indicator)
    
    ax.set_title(title)
    ax.set_ylabel('Value')
    ax.grid(True)
    ax.legend()
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_bollinger_bands(
    df: pd.DataFrame,
    title: str = "Bollinger Bands",
    figsize: tuple = (12, 6)
) -> None:
    """
    Plot price with Bollinger Bands.
    
    Args:
        df (pd.DataFrame): DataFrame with price and Bollinger Bands data
        title (str, optional): Plot title. Defaults to "Bollinger Bands"
        figsize (tuple, optional): Figure size. Defaults to (12, 6)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot price and bands
    ax.plot(df['timestamp'], df['close'], label='Close Price')
    ax.plot(df['timestamp'], df['bb_upper'], 'r--', label='Upper Band')
    ax.plot(df['timestamp'], df['bb_middle'], 'g--', label='Middle Band')
    ax.plot(df['timestamp'], df['bb_lower'], 'r--', label='Lower Band')
    
    ax.set_title(title)
    ax.set_ylabel('Price (USDT)')
    ax.grid(True)
    ax.legend()
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_macd(
    df: pd.DataFrame,
    title: str = "MACD",
    figsize: tuple = (12, 8)
) -> None:
    """
    Plot MACD indicator.
    
    Args:
        df (pd.DataFrame): DataFrame with MACD data
        title (str, optional): Plot title. Defaults to "MACD"
        figsize (tuple, optional): Figure size. Defaults to (12, 8)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot price
    ax1.plot(df['timestamp'], df['close'], label='Close Price')
    ax1.set_title(title)
    ax1.set_ylabel('Price (USDT)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot MACD
    ax2.plot(df['timestamp'], df['macd'], label='MACD')
    ax2.plot(df['timestamp'], df['macd_signal'], label='Signal')
    ax2.bar(df['timestamp'], df['macd_histogram'], label='Histogram')
    ax2.set_ylabel('MACD')
    ax2.grid(True)
    ax2.legend()
    
    # Format x-axis
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show() 