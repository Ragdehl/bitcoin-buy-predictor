from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

class BacktestResult:
    """
    Class to store and analyze backtest results.
    """
    def __init__(
        self,
        trades: List[Dict],
        initial_capital: float,
        final_capital: float,
        df: pd.DataFrame
    ):
        """
        Initialize BacktestResult.
        
        Args:
            trades (List[Dict]): List of trade dictionaries
            initial_capital (float): Initial capital
            final_capital (float): Final capital
            df (pd.DataFrame): DataFrame with price and signal data
        """
        self.trades = trades
        self.initial_capital = initial_capital
        self.final_capital = final_capital
        self.df = df
        
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Returns:
            Dict[str, float]: Dictionary with performance metrics
        """
        if not self.trades:
            return {
                'total_return': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0
            }
        
        # Calculate returns
        returns = [trade['profit'] for trade in self.trades]
        winning_trades = [r for r in returns if r > 0]
        losing_trades = [r for r in returns if r < 0]
        
        # Calculate metrics
        total_return = (self.final_capital - self.initial_capital) / self.initial_capital
        win_rate = len(winning_trades) / len(returns) if returns else 0
        
        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Maximum drawdown
        cumulative_returns = np.cumsum(returns)
        max_drawdown = 0
        peak = cumulative_returns[0]
        
        for value in cumulative_returns:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown
        }

def run_backtest(
    df: pd.DataFrame,
    initial_capital: float,
    position_size: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None
) -> BacktestResult:
    """
    Run backtest simulation.
    
    Args:
        df (pd.DataFrame): DataFrame with price and signal data
        initial_capital (float): Initial capital
        position_size (float): Position size as fraction of capital
        stop_loss (Optional[float]): Stop loss percentage
        take_profit (Optional[float]): Take profit percentage
        
    Returns:
        BacktestResult: Backtest results
    """
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    
    for i in range(1, len(df)):
        current_price = df['close'].iloc[i]
        signal = df['signal'].iloc[i]
        
        # Check for stop loss or take profit if in position
        if position != 0:
            price_change = (current_price - entry_price) / entry_price
            
            if stop_loss and price_change <= -stop_loss:
                # Stop loss hit
                profit = position * (current_price - entry_price)
                capital += profit
                trades.append({
                    'entry_time': df['timestamp'].iloc[i-position],
                    'exit_time': df['timestamp'].iloc[i],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'profit': profit,
                    'exit_reason': 'stop_loss'
                })
                position = 0
                
            elif take_profit and price_change >= take_profit:
                # Take profit hit
                profit = position * (current_price - entry_price)
                capital += profit
                trades.append({
                    'entry_time': df['timestamp'].iloc[i-position],
                    'exit_time': df['timestamp'].iloc[i],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'profit': profit,
                    'exit_reason': 'take_profit'
                })
                position = 0
        
        # Check for new signals
        if signal == 1 and position == 0:  # Buy signal
            position = (capital * position_size) / current_price
            entry_price = current_price
            
        elif signal == -1 and position > 0:  # Sell signal
            profit = position * (current_price - entry_price)
            capital += profit
            trades.append({
                'entry_time': df['timestamp'].iloc[i-position],
                'exit_time': df['timestamp'].iloc[i],
                'entry_price': entry_price,
                'exit_price': current_price,
                'profit': profit,
                'exit_reason': 'signal'
            })
            position = 0
    
    # Close any open position at the end
    if position > 0:
        profit = position * (df['close'].iloc[-1] - entry_price)
        capital += profit
        trades.append({
            'entry_time': df['timestamp'].iloc[-position],
            'exit_time': df['timestamp'].iloc[-1],
            'entry_price': entry_price,
            'exit_price': df['close'].iloc[-1],
            'profit': profit,
            'exit_reason': 'end_of_data'
        })
    
    return BacktestResult(trades, initial_capital, capital, df)

def generate_signals(
    df: pd.DataFrame,
    rsi_oversold: float = 30,
    rsi_overbought: float = 70,
    macd_signal_threshold: float = 0
) -> pd.DataFrame:
    """
    Generate trading signals based on technical indicators.
    
    Args:
        df (pd.DataFrame): DataFrame with price and indicator data
        rsi_oversold (float, optional): RSI oversold threshold. Defaults to 30
        rsi_overbought (float, optional): RSI overbought threshold. Defaults to 70
        macd_signal_threshold (float, optional): MACD signal threshold. Defaults to 0
        
    Returns:
        pd.DataFrame: DataFrame with added signal column
    """
    df = df.copy()
    df['signal'] = 0
    
    # RSI signals
    df.loc[df['rsi'] < rsi_oversold, 'signal'] = 1  # Buy signal
    df.loc[df['rsi'] > rsi_overbought, 'signal'] = -1  # Sell signal
    
    # MACD signals
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        df.loc[df['macd'] > df['macd_signal'] + macd_signal_threshold, 'signal'] = 1
        df.loc[df['macd'] < df['macd_signal'] - macd_signal_threshold, 'signal'] = -1
    
    return df 