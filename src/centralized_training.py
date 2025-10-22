print("--- RUNNING NEW VERSION OF centralized_training.py ---")
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import joblib, json, os
import numpy as np
from imblearn.over_sampling import SMOTE

def get_results_directory():
    """Get the unified results directory"""
    results_dir = os.environ.get('CIRA_RESULTS_DIR', '../results')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def get_models_directory():
    """Get the unified models directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir

def detect_problem_type(y):
    """Detect if problem is classification or regression based on target variable"""
    unique_values = np.unique(y)
    
    # If few unique values and they are integers, treat as classification
    if len(unique_values) <= 10 and all(isinstance(val, (int, np.integer)) for val in unique_values):
        return "classification"
    else:
        return "regression"

def clean_data_for_training(X, y):
    """Clean and validate data before training to prevent common errors"""
    print(f"[DEBUG] Initial data shape: {X.shape}")
    
    # 1. Remove duplicate columns
    X = X.loc[:, ~X.columns.duplicated()]
    print(f"[DEBUG] After removing duplicates: {X.shape}")
    
    # 2. Ensure all data is numeric
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        print(f"[WARN] Dropping {len(non_numeric_cols)} non-numeric columns: {list(non_numeric_cols)}")
        X = X.drop(columns=non_numeric_cols)
    
    # 3. Handle infinite values
    for col in X.columns:
        if X[col].dtype in [np.float64, np.float32]:
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mean())
    
    # 4. Check for constant columns and remove them
    constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_cols:
        print(f"[WARN] Removing {len(constant_cols)} constant columns: {constant_cols}")
        X = X.drop(columns=constant_cols)
    
    # 5. Validate that we still have data
    if X.empty:
        raise ValueError("No features remaining after data cleaning!")
    
    print(f"[DEBUG] Final cleaned data shape: {X.shape}")
    return X, y

def handle_class_imbalance(X, y):
    """Handle class imbalance in the dataset"""
    class_distribution = pd.Series(y).value_counts()
    print(f"[INFO] Original class distribution: {class_distribution.to_dict()}")
    
    if len(class_distribution) <= 1:
        print("[WARN] Only one class found. Cannot balance.")
        return X, y
    
    imbalance_ratio = class_distribution.min() / class_distribution.max()
    print(f"[INFO] Class imbalance ratio: {imbalance_ratio:.4f}")
    
    if imbalance_ratio > 0.3:
        print("[INFO] Class imbalance is acceptable.")
        return X, y
    
    # Apply SMOTE for class balancing
    if len(class_distribution) == 2:
        print("[INFO] Applying SMOTE for class balancing...")
        try:
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            print(f"[INFO] Balanced class distribution: {pd.Series(y_balanced).value_counts().to_dict()}")
            return X_balanced, y_balanced
        except Exception as e:
            print(f"[WARN] SMOTE failed: {e}. Using original data.")
            return X, y
    else:
        print("[INFO] Multi-class problem, using class weights in models.")
        return X, y

def train_models(X, y):
    """Train centralized ML models for risk classification/regression"""
    # --- Data Cleaning First ---
    X_clean, y_clean = clean_data_for_training(X, y)
    
    # --- Get unified directories ---
    results_dir = get_results_directory()
    models_dir = get_models_directory()
    
    print(f"[INFO] Models will be saved to: {models_dir}")
    print(f"[INFO] Metrics will be saved to: {results_dir}")
    
    # --- Detect problem type ---
    problem_type = detect_problem_type(y_clean)
    
    # --- Ensure proper class labels for classification ---
    if problem_type == "classification":
        # Ensure classes are 0, 1, 2, ... for proper classification
        unique_classes = np.unique(y_clean)
        if not np.array_equal(unique_classes, np.arange(len(unique_classes))):
            print(f"[INFO] Remapping classes from {unique_classes.tolist()} to sequential integers")
            class_mapping = {orig: new for new, orig in enumerate(unique_classes)}
            y_clean = y_clean.map(class_mapping)
            print(f"[INFO] New class distribution: {pd.Series(y_clean).value_counts().to_dict()}")
        
        # Handle class imbalance
        X_clean, y_clean = handle_class_imbalance(X_clean, y_clean)
    
    print(f"[INFO] Detected problem type: {problem_type}")
    
    # --- Data Splitting ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, 
        stratify=y_clean if problem_type == "classification" else None
    )
    
    print(f"[INFO] Training set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"[INFO] Target stats - Min: {y_clean.min():.4f}, Max: {y_clean.max():.4f}, Mean: {y_clean.mean():.4f}")
    
    if problem_type == "classification":
        print(f"[INFO] Class distribution: {pd.Series(y_clean).value_counts().to_dict()}")

    # --- Model Configuration ---
    if problem_type == "classification":
        models = {
            "xgboost_model.pkl": XGBClassifier(
                eval_metric="logloss",
                random_state=42,
                n_estimators=100
            ),
            "lightgbm_model.pkl": LGBMClassifier(
                verbosity=-1,
                random_state=42,
                n_estimators=100
            ),
            "random_forest_model.pkl": RandomForestClassifier(
                random_state=42,
                n_estimators=100
            )
        }
    else:
        models = {
            "xgboost_regressor.pkl": XGBRegressor(
                random_state=42,
                n_estimators=100
            ),
            "lightgbm_regressor.pkl": LGBMRegressor(
                random_state=42,
                n_estimators=100,
                verbosity=-1
            ),
            "random_forest_regressor.pkl": RandomForestRegressor(
                random_state=42,
                n_estimators=100
            )
        }

    results = {}
    
    # --- Model Training Loop ---
    for name, model in models.items():
        try:
            print(f"[INFO] Training {name}...")
            
            model.fit(X_train, y_train)
            
            if problem_type == "classification":
                preds = model.predict(X_test)
                preds_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2 else preds
                
                # Calculate classification metrics
                metrics = {
                    "accuracy": accuracy_score(y_test, preds),
                    "f1": f1_score(y_test, preds, average='weighted'),
                    "roc_auc": roc_auc_score(y_test, preds_proba) if len(np.unique(y_test)) == 2 else 0.5
                }
            else:
                preds = model.predict(X_test)
                
                # Calculate regression metrics
                metrics = {
                    "mse": mean_squared_error(y_test, preds),
                    "rmse": np.sqrt(mean_squared_error(y_test, preds)),
                    "r2": r2_score(y_test, preds),
                    "mae": np.mean(np.abs(y_test - preds))
                }
            
            results[name] = metrics
            
            # Save model to unified models directory
            model_path = os.path.join(models_dir, name)
            joblib.dump(model, model_path)
            
            if problem_type == "classification":
                print(f"[SUCCESS] {name} trained and saved. Accuracy: {metrics['accuracy']:.4f}")
            else:
                print(f"[SUCCESS] {name} trained and saved. R2: {metrics['r2']:.4f}")
            
        except Exception as e:
            print(f"[ERROR] Failed to train {name}: {e}")
            results[name] = {"error": str(e)}
            continue

    # --- Save Results to unified results directory ---
    print(f"[INFO] Saving metrics to: {results_dir}")
    
    # Debugging: Print the results dictionary before saving
    print("[DEBUG] Final results dictionary:")
    print(json.dumps(results, indent=4))

    # Robust file writing to unified results directory
    file_path = os.path.join(results_dir, "metrics_report.json")
    print(f"[DEBUG] Attempting to write to: {os.path.abspath(file_path)}")
    try:
        with open(file_path, "w") as f:
            json.dump(results, f, indent=4)
            f.flush()
        print("[SUCCESS] Metrics report saved successfully.")
        
        # Also save a human-readable version
        txt_file_path = os.path.join(results_dir, "metrics_summary.txt")
        with open(txt_file_path, "w") as f:
            f.write("Centralized Training Results\n")
            f.write("=" * 30 + "\n")
            f.write(f"Problem Type: {problem_type}\n")
            f.write(f"Training Samples: {X_train.shape[0]}\n")
            f.write(f"Test Samples: {X_test.shape[0]}\n")
            f.write(f"Features: {X_train.shape[1]}\n\n")
            
            for model_name, metrics in results.items():
                f.write(f"{model_name}:\n")
                if "error" in metrics:
                    f.write(f"  ERROR: {metrics['error']}\n")
                else:
                    for metric_name, value in metrics.items():
                        f.write(f"  {metric_name}: {value:.4f}\n")
                f.write("\n")
        print("[SUCCESS] Metrics summary saved.")
        
    except Exception as e:
        print(f"[ERROR] Failed to write metrics file: {e}")

    print("[INFO] Centralized training complete.")
    
    # Return results for potential use in other modules
    return results, problem_type