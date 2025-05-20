import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timedelta

def load_feature_config(config_path: str = 'features/features_config.json') -> Dict[str, Dict[str, Any]]:
    """Load feature configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def get_feature_groups(config: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """Group features by their type for better organization in the UI."""
    groups = {
        'Price': ['close', 'open', 'high', 'low'],
        'Volume': ['volume'],
        'Trend': [],
        'Momentum': [],
        'Volatility': [],
        'Candle Shape': [],
        'Candle Patterns': []
    }
    
    for feature_name, feature_config in config.items():
        if not feature_config.get('enabled', True):
            continue
            
        feature_type = feature_config['type']
        if feature_type == 'EMA':
            groups['Trend'].append(feature_name)
        elif feature_type == 'RSI':
            groups['Momentum'].append(feature_name)
        elif feature_type == 'BollingerBands':
            groups['Volatility'].append(feature_name)
        elif feature_type == 'CandleShape':
            groups['Candle Shape'].append(feature_name)
        elif feature_type == 'CandlePatterns':
            groups['Candle Patterns'].append(feature_name)
    
    # Remove empty groups
    return {k: v for k, v in groups.items() if v}

def create_hover_template(selected_attributes: List[str], df: pd.DataFrame) -> str:
    """Create a hover template with selected attributes."""
    template_parts = ["Time: %{x}"]
    
    for attr in selected_attributes:
        if attr in df.columns:
            if attr in ['open', 'high', 'low', 'close']:
                template_parts.append(f"{attr.capitalize()}: %{{{attr}:.2f}}")
            else:
                template_parts.append(f"{attr}: %{{{attr}:.2f}}")
    
    template_parts.append("<extra></extra>")
    return "<br>".join(template_parts)

def plot_price_and_volume(df: pd.DataFrame, selected_features: List[str], date_range: tuple, hover_attributes: List[str]) -> None:
    """Plot price and volume with selected features."""
    # Filter data by date range
    mask = (df['timestamp'] >= date_range[0]) & (df['timestamp'] <= date_range[1])
    df_filtered = df[mask].copy()
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3]
    )
    
    # Add price
    fig.add_trace(
        go.Candlestick(
            x=df_filtered['timestamp'],
            open=df_filtered['open'],
            high=df_filtered['high'],
            low=df_filtered['low'],
            close=df_filtered['close'],
            name='Price',
            hoverinfo='all',
            hoverlabel=dict(
                bgcolor='white',
                font_size=12,
                font_family='Rockwell'
            )
        ),
        row=1, col=1
    )
    
    # Add Bollinger Bands if selected
    bb_features = [f for f in selected_features if f.startswith('BB_')]
    if bb_features:
        for feature in bb_features:
            if feature in df_filtered.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_filtered['timestamp'],
                        y=df_filtered[feature],
                        name=feature,
                        line=dict(width=1),
                        hovertemplate=create_hover_template(hover_attributes, df_filtered)
                    ),
                    row=1, col=1
                )
    
    # Add other selected features
    other_features = [f for f in selected_features if not f.startswith('BB_')]
    for feature in other_features:
        if feature in df_filtered.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_filtered['timestamp'],
                    y=df_filtered[feature],
                    name=feature,
                    line=dict(width=1),
                    hovertemplate=create_hover_template(hover_attributes, df_filtered)
                ),
                row=1, col=1
            )
    
    # Add volume
    fig.add_trace(
        go.Bar(
            x=df_filtered['timestamp'],
            y=df_filtered['volume'],
            name='Volume',
            marker_color='rgba(0,0,255,0.3)',
            hovertemplate=create_hover_template(hover_attributes, df_filtered)
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='Price and Volume',
        yaxis_title='Price',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False,
        height=800,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_technical_indicators(df: pd.DataFrame, selected_features: List[str], date_range: tuple, hover_attributes: List[str]) -> None:
    """Plot technical indicators in appropriate subplots based on their type."""
    # Filter data by date range
    mask = (df['timestamp'] >= date_range[0]) & (df['timestamp'] <= date_range[1])
    df_filtered = df[mask].copy()
    
    # Group features by type
    config = load_feature_config()
    feature_groups = get_feature_groups(config)
    
    # Create subplots based on selected feature types
    selected_types = set()
    for feature in selected_features:
        for type_name, features in feature_groups.items():
            if feature in features:
                selected_types.add(type_name)
    
    # Create figure with appropriate number of subplots
    fig = make_subplots(
        rows=len(selected_types) + 1,  # +1 for price
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=['Price'] + list(selected_types)
    )
    
    # Add price
    fig.add_trace(
        go.Candlestick(
            x=df_filtered['timestamp'],
            open=df_filtered['open'],
            high=df_filtered['high'],
            low=df_filtered['low'],
            close=df_filtered['close'],
            name='Price',
            hoverinfo='all',
            hoverlabel=dict(
                bgcolor='white',
                font_size=12,
                font_family='Rockwell'
            )
        ),
        row=1, col=1
    )
    
    # Add features to appropriate subplots
    current_row = 2
    for type_name in selected_types:
        type_features = [f for f in selected_features if f in feature_groups[type_name]]
        
        for feature in type_features:
            if feature in df_filtered.columns:
                # Special handling for RSI
                if feature.startswith('RSI'):
                    yaxis_range = [0, 100]
                else:
                    yaxis_range = None
                
                fig.add_trace(
                    go.Scatter(
                        x=df_filtered['timestamp'],
                        y=df_filtered[feature],
                        name=feature,
                        line=dict(width=1),
                        hovertemplate=create_hover_template(hover_attributes, df_filtered)
                    ),
                    row=current_row, col=1
                )
                
                # Update y-axis range for RSI
                if yaxis_range:
                    fig.update_yaxes(range=yaxis_range, row=current_row, col=1)
        
        current_row += 1
    
    # Update layout
    fig.update_layout(
        title='Technical Indicators',
        height=300 * (len(selected_types) + 1),
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_candle_patterns(df: pd.DataFrame, selected_features: List[str], date_range: tuple, hover_attributes: List[str]) -> None:
    """Plot candlestick patterns with markers for pattern occurrences."""
    # Filter data by date range
    mask = (df['timestamp'] >= date_range[0]) & (df['timestamp'] <= date_range[1])
    df_filtered = df[mask].copy()
    
    # Create figure
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df_filtered['timestamp'],
            open=df_filtered['open'],
            high=df_filtered['high'],
            low=df_filtered['low'],
            close=df_filtered['close'],
            name='Price',
            hoverinfo='all',
            hoverlabel=dict(
                bgcolor='white',
                font_size=12,
                font_family='Rockwell'
            )
        )
    )
    
    # Add markers for pattern occurrences
    for feature in selected_features:
        if feature in df_filtered.columns:
            pattern_dates = df_filtered[df_filtered[feature] == 1]['timestamp']
            pattern_prices = df_filtered[df_filtered[feature] == 1]['high'] * 1.01  # Slightly above the high
            
            fig.add_trace(
                go.Scatter(
                    x=pattern_dates,
                    y=pattern_prices,
                    mode='markers',
                    name=feature,
                    marker=dict(
                        symbol='triangle-down',
                        size=10,
                        color='red'
                    ),
                    hovertemplate=create_hover_template(hover_attributes, df_filtered)
                )
            )
    
    # Update layout
    fig.update_layout(
        title='Candlestick Patterns',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=600,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main Streamlit application for visualizing Bitcoin data and features."""
    st.set_page_config(page_title="Bitcoin Analysis", layout="wide")
    st.title("Bitcoin Price Analysis")
    
    # Load data
    try:
        df = pd.read_csv('data/historical.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    # Load feature configuration
    try:
        config = load_feature_config()
        feature_groups = get_feature_groups(config)
    except Exception as e:
        st.error(f"Error loading feature configuration: {str(e)}")
        return
    
    # Date range selector
    min_date = df['timestamp'].min()
    max_date = df['timestamp'].max()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            min_date,
            min_value=min_date,
            max_value=max_date
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            max_date,
            min_value=min_date,
            max_value=max_date
        )
    
    date_range = (pd.Timestamp(start_date), pd.Timestamp(end_date))
    
    # Hover attributes selection
    st.subheader("Hover Information")
    available_attributes = sorted(df.columns.tolist())
    hover_attributes = st.multiselect(
        "Select attributes to show in hover",
        available_attributes,
        default=['open', 'high', 'low', 'close', 'volume']
    )
    
    # Feature selection
    st.subheader("Feature Selection")
    
    # Create tabs for different feature groups
    tabs = st.tabs(list(feature_groups.keys()))
    
    selected_features = []
    for tab, (group_name, features) in zip(tabs, feature_groups.items()):
        with tab:
            if features:  # Only show tab if there are features in the group
                selected = st.multiselect(
                    f"Select {group_name} Features",
                    features,
                    key=f"select_{group_name}"
                )
                selected_features.extend(selected)
    
    # Debug information
    st.write("Debug Information:")
    st.write("Available Features:", list(feature_groups.keys()))
    st.write("Selected Features:", selected_features)
    st.write("Hover Attributes:", hover_attributes)
    
    # Plot selected features
    if selected_features:
        st.subheader("Visualization")
        
        # Create tabs for different visualization types
        viz_tabs = st.tabs(["Price & Volume", "Technical Indicators", "Candle Patterns"])
        
        with viz_tabs[0]:
            plot_price_and_volume(df, selected_features, date_range, hover_attributes)
        
        with viz_tabs[1]:
            plot_technical_indicators(df, selected_features, date_range, hover_attributes)
        
        with viz_tabs[2]:
            # Filter for only candle pattern features
            pattern_features = [f for f in selected_features if f in feature_groups.get('Candle Patterns', [])]
            if pattern_features:
                plot_candle_patterns(df, pattern_features, date_range, hover_attributes)
            else:
                st.info("No candle pattern features selected")
    else:
        st.info("Please select features to visualize")

if __name__ == "__main__":
    main()
