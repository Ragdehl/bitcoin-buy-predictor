from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def prepare_features(
    df: pd.DataFrame,
    target_column: str = 'signal',
    feature_columns: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features and target for model training.
    
    Args:
        df (pd.DataFrame): DataFrame with features and target
        target_column (str, optional): Name of target column. Defaults to 'signal'
        feature_columns (Optional[List[str]], optional): List of feature columns. Defaults to None
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Features and target arrays
    """
    if feature_columns is None:
        # Use all numeric columns except timestamp and target
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in feature_columns if col not in ['timestamp', target_column]]
    
    X = df[feature_columns].values
    y = df[target_column].values
    
    return X, y

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[RandomForestClassifier, StandardScaler, Dict[str, float]]:
    """
    Train a Random Forest model.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        test_size (float, optional): Test set size. Defaults to 0.2
        random_state (int, optional): Random seed. Defaults to 42
        
    Returns:
        Tuple[RandomForestClassifier, StandardScaler, Dict[str, float]]: Trained model, scaler, and metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state
    )
    model.fit(X_train_scaled, y_train)
    
    # Calculate metrics
    y_pred = model.predict(X_test_scaled)
    metrics = {
        'accuracy': model.score(X_test_scaled, y_test),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return model, scaler, metrics

def predict_signals(
    model: RandomForestClassifier,
    scaler: StandardScaler,
    df: pd.DataFrame,
    feature_columns: List[str]
) -> pd.DataFrame:
    """
    Generate trading signals using trained model.
    
    Args:
        model (RandomForestClassifier): Trained model
        scaler (StandardScaler): Fitted scaler
        df (pd.DataFrame): DataFrame with features
        feature_columns (List[str]): List of feature columns
        
    Returns:
        pd.DataFrame: DataFrame with added prediction column
    """
    df = df.copy()
    
    # Prepare features
    X = df[feature_columns].values
    X_scaled = scaler.transform(X)
    
    # Generate predictions
    df['predicted_signal'] = model.predict(X_scaled)
    
    return df

def evaluate_predictions(
    df: pd.DataFrame,
    actual_column: str = 'signal',
    predicted_column: str = 'predicted_signal'
) -> Dict[str, float]:
    """
    Evaluate prediction performance.
    
    Args:
        df (pd.DataFrame): DataFrame with actual and predicted signals
        actual_column (str, optional): Name of actual signal column. Defaults to 'signal'
        predicted_column (str, optional): Name of predicted signal column. Defaults to 'predicted_signal'
        
    Returns:
        Dict[str, float]: Dictionary with evaluation metrics
    """
    # Calculate accuracy
    accuracy = (df[actual_column] == df[predicted_column]).mean()
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(df[actual_column], df[predicted_column])
    
    # Calculate precision and recall for each class
    report = classification_report(
        df[actual_column],
        df[predicted_column],
        output_dict=True
    )
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': report
    } 