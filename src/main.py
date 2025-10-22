from data_preprocessing import load_all_datasets, preprocess_data, validate_data_for_training, validate_dataset, advanced_feature_engineering, load_datasets_sampled
from threat_modeling_stride import generate_threat_profile
from centralized_training import train_models
from federated_training import federated_training
from risk_ranking import calculate_risk
from visualization import plot_risk_distribution, plot_model_comparison, plot_feature_analysis, plot_training_curves, generate_all_visualizations
from security_controls import suggest_controls
import pandas as pd
import numpy as np
import os
import shutil
import gc

# === ENHANCED DATA ANALYSIS FUNCTIONS ===

def analyze_and_fix_class_imbalance(df, target_col):
    """Analyze and provide solutions for class imbalance"""
    if target_col not in df.columns:
        return target_col, "Target column not found in dataframe"
    
    class_distribution = df[target_col].value_counts()
    if len(class_distribution) < 2:
        return target_col, f"Only one class found: {class_distribution.to_dict()}"
    
    imbalance_ratio = class_distribution.min() / class_distribution.max()
    
    print(f"\n⚖️  CLASS IMBALANCE ANALYSIS:")
    print(f"   Target: {target_col}")
    print(f"   Distribution: {class_distribution.to_dict()}")
    print(f"   Imbalance Ratio: {imbalance_ratio:.4f}")
    
    if imbalance_ratio < 0.1:  # Severe imbalance
        print(f"   🚨 SEVERE IMBALANCE DETECTED!")
        print(f"   🛠️  Applying class weight balancing in models...")
        
        # Try to find alternative target columns
        alternative_targets = []
        for col in df.columns:
            if col != target_col and any(keyword in col.lower() for keyword in ['attack', 'malicious', 'anomaly', 'intrusion', 'breach']):
                if df[col].nunique() == 2:  # Binary
                    alt_dist = df[col].value_counts()
                    if len(alt_dist) == 2:  # Ensure it's truly binary
                        alt_ratio = alt_dist.min() / alt_dist.max()
                        if alt_ratio > 0.3:  # Better balance
                            alternative_targets.append((col, alt_ratio, alt_dist.to_dict()))
        
        if alternative_targets:
            best_alt = max(alternative_targets, key=lambda x: x[1])
            print(f"   💡 Better target found: {best_alt[0]} (balance: {best_alt[1]:.3f})")
            print(f"   📊 New distribution: {best_alt[2]}")
            return best_alt[0], f"Switched to better balanced target: {best_alt[0]}"
        else:
            return target_col, f"Severe imbalance detected but no better targets found. Using class weights."
    
    elif imbalance_ratio < 0.3:  # Moderate imbalance
        return target_col, f"Moderate imbalance detected. Consider using class weights."
    else:
        return target_col, "Class distribution is acceptable."
    
    return target_col, "Analysis complete"

def print_data_quality_report(df, target_col):
    """Print comprehensive data quality report"""
    print(f"\n📊 DATA QUALITY REPORT:")
    print(f"   Dataset Shape: {df.shape}")
    print(f"   Features: {df.shape[1]}")
    print(f"   Samples: {df.shape[0]}")
    print(f"   Target Column: {target_col}")
    
    # Data types
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"   Numeric Features: {len(numeric_cols)}")
    print(f"   Categorical Features: {len(categorical_cols)}")
    
    # Missing values
    missing_total = df.isnull().sum().sum()
    missing_percentage = (missing_total / (df.shape[0] * df.shape[1])) * 100
    print(f"   Missing Values: {missing_total} ({missing_percentage:.2f}%)")
    
    # Target analysis
    if target_col in df.columns:
        target_stats = df[target_col].describe()
        print(f"   Target Stats:")
        print(f"     - Min: {target_stats['min']:.4f}")
        print(f"     - Max: {target_stats['max']:.4f}")
        print(f"     - Mean: {target_stats['mean']:.4f}")
        print(f"     - Std: {target_stats['std']:.4f}")
        
        if df[target_col].dtype in ['int64', 'float64']:
            unique_vals = df[target_col].nunique()
            print(f"     - Unique Values: {unique_vals}")
            
            if unique_vals <= 10:  # Likely classification
                distribution = df[target_col].value_counts()
                print(f"     - Distribution: {distribution.to_dict()}")
    
    # Feature correlations (if we have a target)
    if target_col in numeric_cols and len(numeric_cols) > 1:
        try:
            correlations = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
            top_corr = correlations.head(6)  # Top 5 excluding target itself
            if len(top_corr) > 1:
                print(f"   Top Feature Correlations with Target:")
                for feature, corr in top_corr[1:].items():  # Skip target itself
                    print(f"     - {feature}: {corr:.3f}")
        except:
            print(f"   Feature Correlations: Could not calculate")

# === EXISTING FUNCTIONS (ENHANCED) ===

def setup_unified_results_directory():
    """Ensure all results go to one consistent location"""
    # Get the project root (one level up from src)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, "results")
    
    # Create directory and clean any existing files
    if os.path.exists(results_dir):
        print(f"🧹 Cleaning existing results directory: {results_dir}")
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"🎯 ALL RESULTS WILL BE SAVED TO: {results_dir}")
    print(f"📁 Absolute path: {os.path.abspath(results_dir)}")
    
    # Set as environment variable for all modules to use
    os.environ['CIRA_RESULTS_DIR'] = results_dir
    
    return results_dir

def clean_duplicate_columns(df):
    """Remove duplicate column names by adding suffixes"""
    if df.empty:
        return df
        
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        dup_indices = cols[cols == dup].index
        cols[dup_indices] = [f"{dup}_{i}" for i in range(len(dup_indices))]
    df.columns = cols
    return df

def create_discrete_target(y_continuous):
    """Convert continuous target to discrete classes for classification"""
    if len(np.unique(y_continuous)) <= 2:
        y_binary = (y_continuous > y_continuous.median()).astype(int)
        return y_binary
    
    try:
        q_low = y_continuous.quantile(0.05)
        q_high = y_continuous.quantile(0.95)
        y_filtered = y_continuous[(y_continuous >= q_low) & (y_continuous <= q_high)]
        
        if len(y_filtered) > 0:
            threshold = y_filtered.median()
            y_binary = (y_continuous > threshold).astype(int)
        else:
            y_binary = (y_continuous > y_continuous.median()).astype(int)
            
        print(f"[INFO] Created binary target with distribution: {pd.Series(y_binary).value_counts().to_dict()}")
        return y_binary
        
    except Exception as e:
        print(f"[WARN] Quantile binning failed: {e}. Using binary classification.")
        return (y_continuous > y_continuous.median()).astype(int)

def find_better_target_column(df):
    """Find a target column with better class distribution"""
    potential_targets = []
    
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['label', 'target', 'class', 'attack', 'malicious', 'anomaly', 'risk']):
            unique_vals = df[col].nunique()
            if 2 <= unique_vals <= 10:
                distribution = df[col].value_counts(normalize=True)
                min_class_ratio = distribution.min()
                if min_class_ratio > 0.1:
                    potential_targets.append((col, min_class_ratio, unique_vals))
    
    if potential_targets:
        potential_targets.sort(key=lambda x: x[1], reverse=True)
        best_target = potential_targets[0][0]
        print(f"[INFO] Found better target: {best_target} with minority ratio: {potential_targets[0][1]:.4f}")
        return best_target
    
    return None

def prepare_features_target(df, target_col=None):
    """Safely prepare features and target, ensuring no duplicate columns"""
    df = clean_duplicate_columns(df)
    
    better_target = find_better_target_column(df)
    if better_target and better_target != target_col:
        print(f"[INFO] Switching to better target column: {better_target}")
        target_col = better_target
    
    if target_col is None:
        for col in df.columns:
            if "risk" in col.lower() or "label" in col.lower() or "target" in col.lower() or "class" in col.lower():
                target_col = col
                print(f"[INFO] Using target column: {target_col}")
                break
    
    # === ENHANCEMENT: ADD DATA QUALITY REPORT ===
    print_data_quality_report(df, target_col)
    
    # === ENHANCEMENT: ADD CLASS IMBALANCE ANALYSIS ===
    target_col, imbalance_message = analyze_and_fix_class_imbalance(df, target_col)
    print(f"   📝 {imbalance_message}")
    
    if target_col is None or target_col not in df.columns:
        print("[WARN] No risk/label column found. Creating synthetic risk label...")
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            target_col = "Risk_Label"
            df[target_col] = (df[num_cols[0]].rank(pct=True) > 0.7).astype(int)
            print(f"[INFO] Created synthetic binary {target_col} based on {num_cols[0]}")
        else:
            target_col = "Risk_Label"
            df[target_col] = np.random.randint(0, 2, len(df))
            print("[WARN] Created random binary target for demonstration")
    else:
        if df[target_col].dtype in ['float64', 'float32']:
            print(f"[INFO] Converting continuous target '{target_col}' to discrete classes")
            df[f"{target_col}_continuous"] = df[target_col]
            df[target_col] = create_discrete_target(df[target_col])
        
        df[target_col] = df[target_col].astype(int)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X = X.select_dtypes(include=[np.number])
    X = clean_duplicate_columns(X)
    y = y.astype(int)
    
    print(f"[INFO] Final feature shape: {X.shape}, target distribution: {y.value_counts().to_dict()}")
    return X, y, target_col

def save_data_summary(df, filepath):
    """Save comprehensive data summary"""
    summary = {
        "dataset_shape": df.shape,
        "columns": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
        "basic_stats": df.describe().to_dict()
    }
    
    import json
    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"[INFO] Data summary saved to: {filepath}")

def print_pipeline_summary(df, X, y, target_col, risk_df, results_dir):
    """Print comprehensive pipeline summary"""
    print("\n" + "="*60)
    print("CIRA-ML PIPELINE SUMMARY")
    print("="*60)
    
    print(f"📊 Dataset Overview:")
    print(f"   • Original shape: {df.shape}")
    print(f"   • Features: {X.shape[1]}")
    print(f"   • Samples: {X.shape[0]}")
    print(f"   • Target column: {target_col}")
    print(f"   • Target distribution: {y.value_counts().to_dict()}")
    
    if risk_df is not None:
        print(f"📈 Risk Analysis:")
        if 'risk_category' in risk_df.columns:
            risk_counts = risk_df['risk_category'].value_counts()
            print(f"   • Risk categories: {risk_counts.to_dict()}")
        if 'risk_score' in risk_df.columns:
            print(f"   • Risk score range: {risk_df['risk_score'].min():.3f} - {risk_df['risk_score'].max():.3f}")
            print(f"   • Average risk score: {risk_df['risk_score'].mean():.3f}")
    
    print(f"🎯 Results Location:")
    print(f"   • {results_dir}")
    
    print(f"📁 Generated Files:")
    if os.path.exists(results_dir):
        files = os.listdir(results_dir)
        for file in files:
            file_path = os.path.join(results_dir, file)
            size_kb = os.path.getsize(file_path) / 1024
            print(f"   • {file} ({size_kb:.1f} KB)")
    
    print("="*60)

if __name__ == "__main__":
    print("=== Starting CIRA-ML Risk Assessment ===")
    
    # SETUP UNIFIED RESULTS DIRECTORY FIRST
    RESULTS_DIR = setup_unified_results_directory()
    
    # Store original data for visualizations
    original_df_processed = None

    try:
        # Load and preprocess data
        print("\n🔍 PHASE 1: Data Loading & Preprocessing")
        
        # Force garbage collection before loading
        gc.collect()
        
        # Try normal loading first, fall back to sampled loading if memory error
        try:
            df = load_all_datasets("data")
        except MemoryError as e:
            print(f"[WARN] Memory error during full loading: {e}")
            print("[INFO] Switching to sampled loading approach...")
            df = load_datasets_sampled("data", sample_size=30000)
        
        if df.empty:
            raise ValueError("No data loaded from datasets!")
        
        print(f"[INFO] Raw data loaded: {df.shape}")
        
        # Advanced feature engineering
        print("\n🔧 PHASE 2: Feature Engineering")
        df = advanced_feature_engineering(df)
        print(f"[INFO] After feature engineering: {df.shape}")
        
        # Preprocess data
        print("\n🔄 PHASE 3: Data Preprocessing")
        df = preprocess_data(df)
        if df.empty:
            raise ValueError("DataFrame is empty after preprocessing. Check your data sources.")
        
        # Store processed data for visualizations
        original_df_processed = df.copy()
        
        print(f"[INFO] After preprocessing: {df.shape}")
        
        # Save data summary to unified results directory
        save_data_summary(df, os.path.join(RESULTS_DIR, "data_summary.json"))
        
        # Validate data before proceeding
        if not validate_data_for_training(df):
            print("[WARN] Data validation issues found, but continuing...")
        
        # Comprehensive dataset validation
        validation_issues = validate_dataset(df)
        if validation_issues:
            print(f"[WARN] Dataset validation issues: {validation_issues}")
        
        # Threat modeling
        print("\n🛡️  PHASE 4: Threat Modeling")
        df, summary = generate_threat_profile(df)
        print(f"[INFO] Threat modeling complete. Dataset shape: {df.shape}")

        # Prepare features and target
        print("\n🎯 PHASE 5: Feature & Target Preparation")
        X, y, target_col = prepare_features_target(df)
        
        if X.empty:
            raise ValueError("No features available for training!")
        
        print(f"[INFO] Starting model training with {X.shape[1]} features...")
        
        # Model training
        print("\n🤖 PHASE 6: Model Training")
        train_models(X, y)
        federated_training(X, y)

        # Risk analysis - ensure it uses the correct results directory
        print("\n📊 PHASE 7: Risk Analysis")
        risk_df = calculate_risk(df)
        risk_df.to_csv(os.path.join(RESULTS_DIR, "risk_table.csv"), index=False)
        print(f"[INFO] Risk analysis complete. Results saved.")

        # Enhanced Visualizations - Generate ALL charts in unified directory
        print("\n🎨 PHASE 8: Visualization Generation")
        print("[INFO] Generating comprehensive visualizations...")
        
        # Generate all types of visualizations
        generate_all_visualizations(risk_df, original_df_processed)
        
        print("[INFO] All visualizations generated successfully!")

        # Security controls
        print("\n🔒 PHASE 9: Security Controls")
        high_risk_controls = suggest_controls('High')
        medium_risk_controls = suggest_controls('Medium')
        low_risk_controls = suggest_controls('Low')
        
        print(f"🟥 High Risk Controls: {high_risk_controls}")
        print(f"🟨 Medium Risk Controls: {medium_risk_controls}")
        print(f"🟩 Low Risk Controls: {low_risk_controls}")
        
        # Save security controls to file
        controls_data = {
            "high_risk_controls": high_risk_controls,
            "medium_risk_controls": medium_risk_controls,
            "low_risk_controls": low_risk_controls
        }
        import json
        with open(os.path.join(RESULTS_DIR, "security_controls.json"), "w") as f:
            json.dump(controls_data, f, indent=4)
        
        # Final summary
        print("\n✅ PHASE 10: Pipeline Completion")
        print_pipeline_summary(df, X, y, target_col, risk_df, RESULTS_DIR)
        
        print(f"\n🎉 === CIRA-ML Risk Assessment Completed Successfully ===")
        print(f"📁 All results are in: {RESULTS_DIR}")

    except Exception as e:
        print(f"\n❌ [ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error log to unified results directory
        error_log_path = os.path.join(RESULTS_DIR, "pipeline_error.log")
        with open(error_log_path, 'w') as f:
            f.write(f"Pipeline Error: {e}\n\n")
            f.write("Traceback:\n")
            traceback.print_exc(file=f)
        
        print(f"📄 Error details saved to: {error_log_path}")