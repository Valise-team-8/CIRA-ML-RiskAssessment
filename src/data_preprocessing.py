import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import gc

def load_all_datasets(folder_path="../data"):
    """Load and merge all CSV datasets in the folder with memory optimization"""
    all_dfs = []
    total_rows = 0
    
    print(f"[INFO] Loading datasets from: {folder_path}")
    
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            path = os.path.join(folder_path, file)
            try:
                # Read with optimized data types to save memory
                df = pd.read_csv(path, low_memory=False)
                print(f"[INFO] Loaded: {file} ({df.shape[0]} rows, {df.shape[1]} columns)")
                
                # Add dataset source as prefix to avoid duplicate column names
                dataset_name = file.replace('.csv', '').replace(' ', '_')[:15]
                df = df.add_prefix(f'{dataset_name}_')
                
                # Optimize memory usage
                df = optimize_memory_usage(df)
                
                all_dfs.append(df)
                total_rows += df.shape[0]
                
                # Clear memory
                del df
                gc.collect()
                
            except Exception as e:
                print(f"[WARN] Skipping {file} due to read error: {e}")

    if not all_dfs:
        print("[ERROR] No datasets were successfully loaded!")
        return pd.DataFrame()
    
    print(f"[INFO] Total rows across all datasets: {total_rows}")
    print(f"[INFO] Merging datasets incrementally...")
    
    # FIXED: Use a copy of the list for iteration
    remaining_dfs = all_dfs.copy()
    merged = remaining_dfs[0]
    
    for i in range(1, len(remaining_dfs)):
        try:
            print(f"[INFO] Merging dataset {i+1}/{len(remaining_dfs)}...")
            merged = pd.concat([merged, remaining_dfs[i]], ignore_index=True, sort=False)
            print(f"[INFO] Current shape: {merged.shape}")
            
            # Optimize memory after each merge
            merged = optimize_memory_usage(merged)
            
            # Clear memory
            del remaining_dfs[i]
            gc.collect()
            
        except MemoryError:
            print(f"[ERROR] Memory error when merging dataset {i+1}. Using sampling approach.")
            # If memory error, sample from remaining datasets
            sampled_dfs = [merged]
            for j in range(i, len(remaining_dfs)):
                # Sample to avoid memory issues
                if len(remaining_dfs[j]) > 10000:
                    sampled_df = remaining_dfs[j].sample(n=10000, random_state=42)
                else:
                    sampled_df = remaining_dfs[j]
                sampled_dfs.append(sampled_df)
            
            merged = pd.concat(sampled_dfs, ignore_index=True, sort=False)
            print(f"[INFO] Used sampling due to memory constraints. Final shape: {merged.shape}")
            break
        except Exception as e:
            print(f"[ERROR] Error merging dataset {i+1}: {e}")
            # Continue with what we have
            break
    
    print(f"[INFO] Final merged dataset shape: {merged.shape}")
    return merged

def optimize_memory_usage(df):
    """Optimize DataFrame memory usage by downcasting numeric types"""
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"[INFO] Memory usage before optimization: {start_mem:.2f} MB")
    
    # Downcast numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        try:
            df[col] = pd.to_numeric(df[col], downcast='integer', errors='ignore')
            df[col] = pd.to_numeric(df[col], downcast='float', errors='ignore')
        except Exception as e:
            print(f"[WARN] Could not optimize column {col}: {e}")
    
    # Convert object columns to category if they have low cardinality
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        try:
            if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
                df[col] = df[col].astype('category')
        except Exception as e:
            print(f"[WARN] Could not convert column {col} to category: {e}")
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"[INFO] Memory usage after optimization: {end_mem:.2f} MB ({((start_mem - end_mem) / start_mem) * 100:.1f}% reduction)")
    
    return df

def load_datasets_sampled(folder_path="../data", sample_size=30000):
    """Load datasets with sampling to avoid memory issues"""
    print(f"[INFO] Using sampled approach with {sample_size} rows per dataset")
    
    sampled_dfs = []
    
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            path = os.path.join(folder_path, file)
            try:
                # Get total rows without loading entire file
                with open(path, 'r') as f:
                    total_rows = sum(1 for line in f) - 1  # Subtract header
                
                print(f"[INFO] Processing {file} ({total_rows} rows)")
                
                # Sample the dataset
                if total_rows > sample_size:
                    # Use chunks to sample large files
                    chunk_size = min(10000, sample_size)
                    chunks = []
                    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
                        chunks.append(chunk)
                        if sum(len(c) for c in chunks) >= sample_size:
                            break
                    
                    sampled_df = pd.concat(chunks, ignore_index=True)
                    sampled_df = sampled_df.sample(n=sample_size, random_state=42)
                else:
                    # Load entire file if it's small
                    sampled_df = pd.read_csv(path, low_memory=False)
                
                # Add prefix and optimize
                dataset_name = file.replace('.csv', '').replace(' ', '_')[:15]
                sampled_df = sampled_df.add_prefix(f'{dataset_name}_')
                sampled_df = optimize_memory_usage(sampled_df)
                
                sampled_dfs.append(sampled_df)
                print(f"[INFO] Sampled: {file} ({len(sampled_df)} rows)")
                
            except Exception as e:
                print(f"[WARN] Skipping {file}: {e}")
    
    if not sampled_dfs:
        raise ValueError("No datasets could be loaded!")
    
    print("[INFO] Merging all sampled datasets...")
    merged = pd.concat(sampled_dfs, ignore_index=True, sort=False)
    print(f"[INFO] Final sampled dataset shape: {merged.shape}")
    return merged

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

def validate_data_for_training(df):
    """Validate that data is ready for ML training"""
    if df.empty:
        print("[ERROR] DataFrame is empty!")
        return False
    
    # Check for duplicate column names
    if df.columns.duplicated().any():
        print("[ERROR] Duplicate column names found!")
        print("Duplicate columns:", df.columns[df.columns.duplicated()].tolist())
        return False
    
    # Check for non-numeric columns
    non_numeric = df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        print(f"[WARN] Found {len(non_numeric)} non-numeric columns: {list(non_numeric)}")
    
    # Check for NaN or infinite values
    if df.isnull().any().any():
        print("[WARN] DataFrame contains NaN values - will be handled in preprocessing")
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number])
    if not numeric_cols.empty:
        if np.isinf(numeric_cols).any().any():
            print("[WARN] DataFrame contains infinite values - will be handled in preprocessing")
    
    print("[INFO] Data validation passed - ready for training")
    return True

def preprocess_data(df: pd.DataFrame):
    """Clean and encode dataset with memory optimization"""
    
    if df.empty:
        print("[WARN] Empty DataFrame received for preprocessing.")
        return pd.DataFrame()
    
    print(f"[INFO] Starting preprocessing with shape: {df.shape}")
    
    # Step 1: Clean duplicate column names FIRST
    df = clean_duplicate_columns(df)
    print(f"[INFO] Column names cleaned. Current shape: {df.shape}")
    
    # Step 2: Handle missing values - drop columns with too many missing values first
    missing_percentage = df.isnull().sum() / len(df)
    cols_to_drop = missing_percentage[missing_percentage > 0.8].index  # Increased threshold
    
    if len(cols_to_drop) > 0:
        print(f"[INFO] Dropping {len(cols_to_drop)} columns with >80% missing values.")
        df = df.drop(columns=cols_to_drop)
    
    if df.empty:
        print("[ERROR] DataFrame is empty after dropping columns!")
        return pd.DataFrame()

    # Step 3: Sample if dataset is too large for processing
    if len(df) > 100000:
        print(f"[INFO] Dataset too large ({len(df)} rows). Sampling to 100000 rows for processing.")
        df = df.sample(n=100000, random_state=42)
        print(f"[INFO] After sampling: {df.shape}")

    # Step 4: Separate numerical and categorical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    print(f"[INFO] Numerical columns: {len(num_cols)}, Categorical columns: {len(cat_cols)}")
    
    # Step 5: Impute missing numerical values
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())
    
    # Step 6: Impute missing categorical values
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
            else:
                df[col] = df[col].fillna('Unknown')
    
    # Step 7: Encode categorical variables - only if not too many categories
    if len(cat_cols) > 0:
        for col in cat_cols:
            try:
                # Skip columns with too many categories
                if df[col].nunique() > 100:
                    print(f"[INFO] Skipping high-cardinality column: {col} ({df[col].nunique()} categories)")
                    df = df.drop(columns=[col])
                    continue
                    
                df[col] = df[col].astype(str)
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col])
                print(f"[INFO] Encoded categorical column: {col}")
            except Exception as e:
                print(f"[WARN] Could not encode column '{col}': {e}")
                df = df.drop(columns=[col])
    
    # Step 8: Handle infinite values and scale numerical data
    num_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(num_cols) > 0:
        for col in num_cols:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
        
        # Scale numerical features
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        print(f"[INFO] Scaled {len(num_cols)} numerical columns")
    
    # Final cleanup: Remove any remaining non-numeric columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        print(f"[INFO] Dropping {len(non_numeric_cols)} non-numeric columns")
        df = df.select_dtypes(include=[np.number])
    
    # Remove constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        print(f"[INFO] Removing {len(constant_cols)} constant columns")
        df = df.drop(columns=constant_cols)
    
    # Final memory optimization
    df = optimize_memory_usage(df)
    
    print(f"[INFO] Preprocessing complete. Final shape: {df.shape}, Columns: {len(df.columns)}")
    
    return df

def validate_dataset(df):
    """Comprehensive data validation"""
    issues = []
    
    if df.empty:
        issues.append("DataFrame is empty")
        return issues
    
    # Check for constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        issues.append(f"Constant columns: {constant_cols}")
    
    # Check for high correlation (only if we have numeric data)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        # Sample for correlation to avoid memory issues
        if len(df) > 10000:
            sample_df = df.sample(n=10000, random_state=42)
        else:
            sample_df = df
            
        corr_matrix = sample_df[numeric_cols].corr()
        high_corr = np.where(np.abs(corr_matrix) > 0.95)
        high_corr_pairs = [(corr_matrix.index[x], corr_matrix.columns[y]) 
                           for x, y in zip(*high_corr) if x != y and x < y]
        if high_corr_pairs:
            issues.append(f"Highly correlated features: {high_corr_pairs[:5]}")
    
    # Check class imbalance if target exists
    target_cols = [col for col in df.columns if 'risk' in col.lower() or 'label' in col.lower()]
    if target_cols:
        target_col = target_cols[0]
        if df[target_col].dtype in ['int64', 'float64']:
            class_balance = df[target_col].value_counts(normalize=True)
            if len(class_balance) > 1 and class_balance.min() < 0.1:
                issues.append(f"Severe class imbalance: {class_balance.to_dict()}")
    
    return issues

def advanced_feature_engineering(df):
    """Create cybersecurity-specific features with memory optimization"""
    print("[INFO] Starting feature engineering...")
    
    # Sample if dataset is too large
    if len(df) > 50000:
        print(f"[INFO] Sampling for feature engineering ({len(df)} -> 50000 rows)")
        df = df.sample(n=50000, random_state=42)
    
    # Network behavior features - only for IP-like columns
    ip_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['ip', 'address'])]
    
    for ip_col in ip_columns[:3]:  # Limit to first 3 IP columns
        try:
            # Only create frequency if reasonable number of unique values
            if df[ip_col].nunique() < 1000:
                freq_col = f"{ip_col}_freq"
                df[freq_col] = df.groupby(ip_col)[ip_col].transform('count')
                print(f"[INFO] Created frequency feature: {freq_col}")
        except Exception as e:
            print(f"[WARN] Could not create frequency feature for {ip_col}: {e}")
    
    print(f"[INFO] Feature engineering complete. Shape: {df.shape}")
    return df

# Test function for debugging
def test_preprocessing():
    """Test the preprocessing pipeline with sample data"""
    print("=== Testing Preprocessing Pipeline ===")
    
    # Create sample data
    sample_data = {
        'Source_IP': ['192.168.1.1', '192.168.1.2', '192.168.1.1', '192.168.1.3'],
        'Destination_IP': ['10.0.0.1', '10.0.0.2', '10.0.0.1', '10.0.0.3'],
        'Protocol': ['TCP', 'UDP', 'TCP', 'HTTP'],
        'Port': [80, 443, 80, 8080],
        'Bytes_Sent': [100, 200, 150, np.nan],
        'Risk_Label': [0, 1, 0, 1]
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Original shape: {df.shape}")
    
    # Test preprocessing
    df_processed = preprocess_data(df)
    print(f"Processed shape: {df_processed.shape}")
    
    # Test validation
    issues = validate_dataset(df_processed)
    if issues:
        print(f"Validation issues: {issues}")
    else:
        print("No validation issues found")
    
    return df_processed

if __name__ == "__main__":
    # Run test if script is executed directly
    test_preprocessing()