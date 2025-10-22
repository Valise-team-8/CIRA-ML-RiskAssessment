import pandas as pd
import numpy as np
import os

def get_results_directory():
    """Get the unified results directory"""
    results_dir = os.environ.get('CIRA_RESULTS_DIR', '../results')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def calculate_risk(df):
    """Calculate comprehensive risk scores with robust error handling"""
    print("[INFO] Calculating risk rankings...")
    
    try:
        # Create a copy to avoid modifying original
        risk_df = df.copy()
        
        # Check if we have a target column for risk calculation
        target_cols = [col for col in risk_df.columns if any(keyword in col.lower() for keyword in ['risk', 'label', 'target', 'attack', 'malicious'])]
        
        if target_cols:
            target_col = target_cols[0]
            print(f"[INFO] Using target column for risk calculation: {target_col}")
            
            # Use target as base risk score
            if risk_df[target_col].dtype in ['int64', 'float64']:
                risk_df['risk_score'] = risk_df[target_col].astype(float)
            else:
                # Convert categorical to numerical
                risk_df['risk_score'] = risk_df[target_col].astype('category').cat.codes
        else:
            print("[WARN] No risk/label column found. Creating synthetic risk scores...")
            # Create synthetic risk scores based on feature patterns
            numeric_cols = risk_df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                # Use first numeric column to create risk scores
                base_col = numeric_cols[0]
                risk_df['risk_score'] = (risk_df[base_col] - risk_df[base_col].min()) / (risk_df[base_col].max() - risk_df[base_col].min())
            else:
                # Fallback: random risk scores
                risk_df['risk_score'] = np.random.uniform(0, 1, len(risk_df))
        
        # Ensure risk_score is properly scaled 0-1
        risk_min = risk_df['risk_score'].min()
        risk_max = risk_df['risk_score'].max()
        
        if risk_max > risk_min:  # Avoid division by zero
            risk_df['risk_score'] = (risk_df['risk_score'] - risk_min) / (risk_max - risk_min)
        else:
            risk_df['risk_score'] = 0.5  # Default medium risk
        
        # Create risk categories
        conditions = [
            risk_df['risk_score'] < 0.33,
            risk_df['risk_score'] < 0.66,
            risk_df['risk_score'] >= 0.66
        ]
        choices = ['Low', 'Medium', 'High']
        risk_df['risk_category'] = np.select(conditions, choices, default='Low')
        
        # Calculate additional risk metrics
        risk_df['risk_percentile'] = risk_df['risk_score'].rank(pct=True)
        
        # Add threat level based on common cybersecurity patterns
        risk_df['threat_level'] = risk_df['risk_score'].apply(
            lambda x: 'Critical' if x > 0.9 else 'High' if x > 0.7 else 'Medium' if x > 0.4 else 'Low'
        )
        
        # Add risk trend (simple moving average if we have enough data)
        if len(risk_df) > 10:
            risk_df['risk_trend'] = risk_df['risk_score'].rolling(window=min(10, len(risk_df)//10), center=True).mean()
        else:
            risk_df['risk_trend'] = risk_df['risk_score']
        
        print(f"[INFO] Risk score range: {risk_df['risk_score'].min():.3f} - {risk_df['risk_score'].max():.3f}")
        print(f"[INFO] Risk categories: {risk_df['risk_category'].value_counts().to_dict()}")
        print(f"[INFO] Threat levels: {risk_df['threat_level'].value_counts().to_dict()}")
        
        # Save risk summary
        results_dir = get_results_directory()
        risk_summary = {
            "total_samples": len(risk_df),
            "risk_score_stats": {
                "mean": float(risk_df['risk_score'].mean()),
                "std": float(risk_df['risk_score'].std()),
                "min": float(risk_df['risk_score'].min()),
                "max": float(risk_df['risk_score'].max())
            },
            "risk_category_distribution": risk_df['risk_category'].value_counts().to_dict(),
            "threat_level_distribution": risk_df['threat_level'].value_counts().to_dict(),
            "high_risk_percentage": float((risk_df['risk_score'] > 0.7).mean() * 100)
        }
        
        import json
        with open(os.path.join(results_dir, "risk_summary.json"), "w") as f:
            json.dump(risk_summary, f, indent=4)
        
        print("[INFO] Risk analysis complete and summary saved.")
        
        return risk_df
        
    except Exception as e:
        print(f"[ERROR] Risk calculation failed: {e}")
        # Return basic dataframe with minimal risk info
        risk_df = df.copy()
        risk_df['risk_score'] = 0.5  # Default medium risk
        risk_df['risk_category'] = 'Medium'
        risk_df['threat_level'] = 'Medium'
        risk_df['risk_percentile'] = 0.5
        
        print("[INFO] Using fallback risk scores due to calculation error.")
        return risk_df

def analyze_risk_patterns(risk_df):
    """Analyze risk patterns and provide insights"""
    print("[INFO] Analyzing risk patterns...")
    
    try:
        insights = []
        
        # Basic risk statistics
        if 'risk_score' in risk_df.columns:
            risk_scores = risk_df['risk_score']
            insights.append(f"Risk Score Analysis:")
            insights.append(f"  • Mean Risk: {risk_scores.mean():.3f}")
            insights.append(f"  • Risk Std Dev: {risk_scores.std():.3f}")
            insights.append(f"  • High Risk (>0.7): {(risk_scores > 0.7).mean()*100:.1f}%")
            insights.append(f"  • Critical Risk (>0.9): {(risk_scores > 0.9).mean()*100:.1f}%")
        
        # Category analysis
        if 'risk_category' in risk_df.columns:
            category_dist = risk_df['risk_category'].value_counts()
            insights.append(f"Risk Categories:")
            for category, count in category_dist.items():
                percentage = (count / len(risk_df)) * 100
                insights.append(f"  • {category}: {count} ({percentage:.1f}%)")
        
        # Threat level analysis
        if 'threat_level' in risk_df.columns:
            threat_dist = risk_df['threat_level'].value_counts()
            insights.append(f"Threat Levels:")
            for level, count in threat_dist.items():
                percentage = (count / len(risk_df)) * 100
                insights.append(f"  • {level}: {count} ({percentage:.1f}%)")
        
        # Save insights
        results_dir = get_results_directory()
        with open(os.path.join(results_dir, "risk_insights.txt"), "w") as f:
            f.write("CIRA-ML RISK ANALYSIS INSIGHTS\n")
            f.write("=" * 50 + "\n\n")
            for insight in insights:
                f.write(insight + "\n")
        
        print("[INFO] Risk pattern analysis complete.")
        return insights
        
    except Exception as e:
        print(f"[ERROR] Risk pattern analysis failed: {e}")
        return ["Risk pattern analysis could not be completed due to errors."]

if __name__ == "__main__":
    # Test the risk calculation
    test_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(1, 2, 100),
        'risk_label': np.random.randint(0, 2, 100)
    })
    
    risk_result = calculate_risk(test_data)
    insights = analyze_risk_patterns(risk_result)
    
    for insight in insights:
        print(insight)