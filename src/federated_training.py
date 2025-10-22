import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

def detect_problem_type(y):
    """Detect if problem is classification or regression"""
    unique_values = np.unique(y)
    if len(unique_values) <= 10 and all(isinstance(val, (int, np.integer)) for val in unique_values):
        return "classification"
    else:
        return "regression"

def federated_training(X, y, n_clients=3):
    """Simple federated learning simulation"""
    print("[INFO] Starting federated training...")
    
    problem_type = detect_problem_type(y)
    print(f"[INFO] Federated training problem type: {problem_type}")
    
    # Ensure y is appropriate type
    if problem_type == "classification":
        y = y.astype(int)
        model_class = RandomForestClassifier
    else:
        model_class = RandomForestRegressor
    
    client_metrics = []
    
    for i in range(n_clients):
        # Simulate data partitioning
        start_idx = i * len(X) // n_clients
        end_idx = (i + 1) * len(X) // n_clients
        
        X_local = X.iloc[start_idx:end_idx]
        y_local = y.iloc[start_idx:end_idx]
        
        # Train local model
        X_train, X_test, y_train, y_test = train_test_split(
            X_local, y_local, test_size=0.2, random_state=42
        )
        
        model = model_class(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if problem_type == "classification":
            accuracy = accuracy_score(y_test, y_pred)
            client_metrics.append(accuracy)
            print(f"[INFO] Client {i+1} local accuracy: {accuracy:.4f}")
        else:
            mse = mean_squared_error(y_test, y_pred)
            client_metrics.append(mse)
            print(f"[INFO] Client {i+1} local MSE: {mse:.4f}")
    
    if problem_type == "classification":
        avg_metric = np.mean(client_metrics)
        print(f"[INFO] Average client accuracy: {avg_metric:.4f}")
    else:
        avg_metric = np.mean(client_metrics)
        print(f"[INFO] Average client MSE: {avg_metric:.4f}")
    
    print("[INFO] Federated training complete.")
    return client_metrics