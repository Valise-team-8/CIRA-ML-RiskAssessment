import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
from matplotlib.gridspec import GridSpec

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def get_results_directory():
    """Get the unified results directory from environment variable"""
    results_dir = os.environ.get('CIRA_RESULTS_DIR', '../results')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def plot_risk_distribution(risk_df):
    """Plot comprehensive risk distribution analysis with robust error handling"""
    print("[INFO] Generating comprehensive risk distribution charts...")
    
    results_dir = get_results_directory()
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig)
    
    # Plot 1: Risk Category Distribution (Pie Chart)
    ax1 = fig.add_subplot(gs[0, 0])
    if 'risk_category' in risk_df.columns and risk_df['risk_category'].nunique() > 1:
        risk_counts = risk_df['risk_category'].value_counts()
        colors = ['#66bb6a', '#ffa726', '#ff6b6b', '#42a5f5']  # Green, Orange, Red, Blue
        wedges, texts, autotexts = ax1.pie(risk_counts.values, labels=risk_counts.index, 
                                          autopct='%1.1f%%', colors=colors[:len(risk_counts)],
                                          startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax1.set_title('Risk Category Distribution', fontsize=14, fontweight='bold', pad=20)
    else:
        ax1.text(0.5, 0.5, 'Insufficient risk category data\nor single category', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=10)
        ax1.set_title('Risk Category Distribution', fontsize=14, fontweight='bold')
    
    # Plot 2: Risk Score Histogram
    ax2 = fig.add_subplot(gs[0, 1])
    if 'risk_score' in risk_df.columns and risk_df['risk_score'].nunique() > 1:
        sns.histplot(data=risk_df, x='risk_score', bins=30, ax=ax2, kde=True, color='skyblue')
        ax2.set_xlabel('Risk Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Risk Score Distribution', fontsize=14, fontweight='bold')
        ax2.axvline(risk_df['risk_score'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {risk_df["risk_score"].mean():.3f}')
        ax2.legend()
        
        # Add statistics
        stats_text = f"Min: {risk_df['risk_score'].min():.3f}\nMax: {risk_df['risk_score'].max():.3f}\nStd: {risk_df['risk_score'].std():.3f}"
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, va='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, 'Insufficient risk score variation', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=10)
        ax2.set_title('Risk Score Distribution', fontsize=14, fontweight='bold')
    
    # Plot 3: Risk Score by Category (Boxplot)
    ax3 = fig.add_subplot(gs[0, 2])
    if 'risk_category' in risk_df.columns and 'risk_score' in risk_df.columns and risk_df['risk_category'].nunique() > 1:
        sns.boxplot(data=risk_df, x='risk_category', y='risk_score', ax=ax3, palette='Set2')
        ax3.set_xlabel('Risk Category')
        ax3.set_ylabel('Risk Score')
        ax3.set_title('Risk Score by Category', fontsize=14, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
    else:
        ax3.text(0.5, 0.5, 'Insufficient data for\ncategory comparison', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=10)
        ax3.set_title('Risk Score by Category', fontsize=14, fontweight='bold')
    
    # Plot 4: Feature Correlation Heatmap
    ax4 = fig.add_subplot(gs[1, :])
    numeric_cols = risk_df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 5:  # Only create heatmap if we have enough features
        try:
            # Sample data if too large for correlation
            if len(risk_df) > 10000:
                sample_df = risk_df.sample(n=10000, random_state=42)
            else:
                sample_df = risk_df
            
            if 'risk_score' in numeric_cols:
                # Get top correlated features with risk score
                correlations = sample_df[numeric_cols].corr()['risk_score'].abs().sort_values(ascending=False)
                top_features = correlations.head(12).index.tolist()  # Top 12 features
                
                if len(top_features) > 2:
                    corr_matrix = sample_df[top_features].corr()
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                    
                    # Create heatmap
                    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                               center=0, ax=ax4, cbar_kws={'shrink': 0.8}, annot_kws={'size': 8})
                    ax4.set_title('Top Feature Correlations with Risk Score', fontsize=14, fontweight='bold', pad=20)
                else:
                    ax4.text(0.5, 0.5, 'Not enough correlated features', 
                            ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                    ax4.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
            else:
                # General correlation matrix
                if len(numeric_cols) > 10:
                    numeric_cols = numeric_cols[:10]  # Limit to first 10 columns
                
                corr_matrix = sample_df[numeric_cols].corr()
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                           center=0, ax=ax4, cbar_kws={'shrink': 0.8}, annot_kws={'size': 8})
                ax4.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        except Exception as e:
            ax4.text(0.5, 0.5, f'Correlation calculation failed:\n{str(e)[:50]}...', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=10)
            ax4.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Not enough numeric features\nfor correlation analysis', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    
    # Plot 5: Threat Level Analysis
    ax5 = fig.add_subplot(gs[2, 0])
    if 'threat_level' in risk_df.columns and risk_df['threat_level'].nunique() > 1:
        threat_counts = risk_df['threat_level'].value_counts()
        colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red', 'Critical': 'darkred'}
        threat_colors = [colors.get(level, 'gray') for level in threat_counts.index]
        
        bars = ax5.bar(threat_counts.index, threat_counts.values, color=threat_colors, alpha=0.7)
        ax5.set_xlabel('Threat Level')
        ax5.set_ylabel('Count')
        ax5.set_title('Threat Level Distribution', fontsize=14, fontweight='bold')
        ax5.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, threat_counts.values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
    else:
        ax5.text(0.5, 0.5, 'Threat level data\nnot available', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=10)
        ax5.set_title('Threat Level Distribution', fontsize=14, fontweight='bold')
    
    # Plot 6: Risk Score Cumulative Distribution
    ax6 = fig.add_subplot(gs[2, 1])
    if 'risk_score' in risk_df.columns and risk_df['risk_score'].nunique() > 1:
        sorted_scores = np.sort(risk_df['risk_score'])
        cum_dist = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        ax6.plot(sorted_scores, cum_dist, linewidth=2, color='purple')
        ax6.set_xlabel('Risk Score')
        ax6.set_ylabel('Cumulative Probability')
        ax6.set_title('Cumulative Risk Distribution', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # Add key percentiles
        percentiles = [25, 50, 75, 90, 95]
        colors = ['red', 'orange', 'yellow', 'green', 'blue']
        for percentile, color in zip(percentiles, colors):
            value = np.percentile(risk_df['risk_score'], percentile)
            ax6.axvline(value, color=color, linestyle='--', alpha=0.7, label=f'{percentile}%')
            ax6.text(value, 0.1, f'{percentile}%', rotation=90, va='bottom', fontsize=8)
        
        ax6.legend(fontsize=8)
    else:
        ax6.text(0.5, 0.5, 'Insufficient risk score data\nfor cumulative distribution', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=10)
        ax6.set_title('Cumulative Risk Distribution', fontsize=14, fontweight='bold')
    
    # Plot 7: Risk Analysis Summary
    ax7 = fig.add_subplot(gs[2, 2])
    
    # Create summary statistics
    summary_stats = []
    
    if 'risk_score' in risk_df.columns:
        summary_stats.append(f"Total Samples: {len(risk_df):,}")
        summary_stats.append(f"Mean Risk: {risk_df['risk_score'].mean():.3f}")
        summary_stats.append(f"Risk Std: {risk_df['risk_score'].std():.3f}")
        summary_stats.append(f"High Risk %: {(risk_df['risk_score'] > 0.7).mean()*100:.1f}%")
    
    if 'risk_category' in risk_df.columns:
        for category, count in risk_df['risk_category'].value_counts().items():
            summary_stats.append(f"{category}: {count} ({count/len(risk_df)*100:.1f}%)")
    
    if summary_stats:
        summary_text = "\n".join(summary_stats)
        ax7.text(0.5, 0.5, summary_text, ha='center', va='center', 
                transform=ax7.transAxes, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    else:
        ax7.text(0.5, 0.5, 'No risk statistics available', 
                ha='center', va='center', transform=ax7.transAxes, fontsize=10)
    
    ax7.set_title('Risk Analysis Summary', fontsize=14, fontweight='bold')
    ax7.axis('off')  # Turn off axes for text display
    
    plt.tight_layout()
    
    # Save to unified results directory
    plt.savefig(os.path.join(results_dir, 'comprehensive_risk_analysis.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("[INFO] Comprehensive risk analysis charts saved.")

def plot_model_comparison():
    """Compare model performance with detailed visualizations and error handling"""
    print("[INFO] Generating model comparison charts...")
    
    results_dir = get_results_directory()
    
    # Load metrics from unified results directory
    metrics_path = os.path.join(results_dir, "metrics_report.json")
    if not os.path.exists(metrics_path):
        print("[WARN] No metrics file found for model comparison")
        # Create a placeholder chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No model metrics available\n\nModels may have failed to train\nor metrics file was not generated', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Model Comparison - No Data Available', fontsize=14, fontweight='bold')
        plt.savefig(os.path.join(results_dir, 'model_comparison_analysis.png'), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        return
    
    try:
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
        
        # Filter out models with errors
        valid_models = {name: metrics for name, metrics in metrics_data.items() if "error" not in metrics}
        
        if not valid_models:
            print("[WARN] No valid model metrics found (all models had errors)")
            # Create error summary chart
            fig, ax = plt.subplots(figsize=(12, 8))
            
            error_summary = "Model Training Errors:\n\n"
            for model_name, metrics in metrics_data.items():
                if "error" in metrics:
                    error_msg = metrics['error']
                    # Truncate long error messages
                    if len(error_msg) > 100:
                        error_msg = error_msg[:100] + "..."
                    error_summary += f"• {model_name.replace('.pkl', '')}: {error_msg}\n"
            
            ax.text(0.5, 0.5, error_summary, ha='center', va='center', 
                    transform=ax.transAxes, fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
            ax.set_title('Model Training Failed - Error Summary', fontsize=14, fontweight='bold')
            ax.axis('off')
            
            plt.savefig(os.path.join(results_dir, 'model_comparison_analysis.png'), 
                        dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            return
        
        # Determine problem type
        problem_type = "classification"
        for model_metrics in valid_models.values():
            if "mse" in model_metrics or "r2" in model_metrics:
                problem_type = "regression"
                break
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Plot 1: Model Performance Comparison
        ax1 = axes[0]
        models = []
        performance_scores = []
        
        for model_name, metrics in valid_models.items():
            models.append(model_name.replace('.pkl', '').replace('_', ' ').title())
            if problem_type == "classification":
                performance_scores.append(metrics.get('accuracy', 0))
            else:
                performance_scores.append(metrics.get('r2', 0))
        
        if models:
            bars = ax1.bar(models, performance_scores, color=sns.color_palette("viridis", len(models)))
            ax1.set_ylabel('Accuracy' if problem_type == 'classification' else 'R² Score')
            ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, score in zip(bars, performance_scores):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No model metrics available', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        
        # Plot 2: Detailed Metrics Comparison
        ax2 = axes[1]
        if problem_type == "classification":
            metric_names = ['accuracy', 'f1', 'roc_auc']
            metric_labels = ['Accuracy', 'F1-Score', 'ROC AUC']
        else:
            metric_names = ['r2', 'mse', 'mae']
            metric_labels = ['R² Score', 'MSE', 'MAE']
        
        metric_data = []
        valid_model_names = []
        
        for model_name, metrics in valid_models.items():
            valid_model_names.append(model_name.replace('.pkl', '').replace('_', ' ').title())
            model_metrics = []
            for metric in metric_names:
                model_metrics.append(metrics.get(metric, 0))
            metric_data.append(model_metrics)
        
        if metric_data:
            metric_data = np.array(metric_data)
            x = np.arange(len(metric_labels))
            width = 0.8 / len(valid_model_names)
            
            for i, model in enumerate(valid_model_names):
                offset = (i - len(valid_model_names)/2 + 0.5) * width
                bars = ax2.bar(x + offset, metric_data[i], width, label=model)
                
                # Add value labels
                for j, value in enumerate(metric_data[i]):
                    ax2.text(x[j] + offset, value + 0.01, f'{value:.3f}', 
                            ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            ax2.set_xlabel('Metrics')
            ax2.set_ylabel('Score')
            ax2.set_title('Detailed Metrics Comparison', fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(metric_labels)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No detailed metrics available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Detailed Metrics Comparison', fontsize=14, fontweight='bold')
        
        # Plot 3: Training Success Analysis
        ax3 = axes[2]
        total_models = len(metrics_data)
        successful_models = len(valid_models)
        failed_models = total_models - successful_models
        
        labels = ['Successful', 'Failed']
        sizes = [successful_models, failed_models]
        colors = ['#66bb6a', '#ff6b6b']
        
        if total_models > 0:
            wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                              startangle=90)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax3.text(0.5, 0.5, 'No model data', ha='center', va='center', transform=ax3.transAxes)
        
        ax3.set_title('Model Training Success Rate', fontsize=14, fontweight='bold')
        
        # Plot 4: Performance Insights
        ax4 = axes[3]
        insights = []
        
        if valid_models:
            # Find best model
            if problem_type == "classification":
                best_model = max(valid_models.items(), key=lambda x: x[1].get('accuracy', 0))
                insights.append(f"Best Model: {best_model[0].replace('.pkl', '')}")
                insights.append(f"Best Accuracy: {best_model[1].get('accuracy', 0):.3f}")
            else:
                best_model = max(valid_models.items(), key=lambda x: x[1].get('r2', 0))
                insights.append(f"Best Model: {best_model[0].replace('.pkl', '')}")
                insights.append(f"Best R²: {best_model[1].get('r2', 0):.3f}")
            
            insights.append(f"Successful Models: {successful_models}/{total_models}")
            insights.append(f"Problem Type: {problem_type.title()}")
        else:
            insights.append("No successful models to analyze")
        
        insight_text = "\n".join(insights)
        ax4.text(0.5, 0.5, insight_text, ha='center', va='center', 
                transform=ax4.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax4.set_title('Performance Insights', fontsize=14, fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Save to unified results directory
        plt.savefig(os.path.join(results_dir, 'model_comparison_analysis.png'), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("[INFO] Model comparison charts saved.")
        
    except Exception as e:
        print(f"[ERROR] Model comparison failed: {e}")
        # Create error chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Error generating model comparison:\n{str(e)}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('Model Comparison - Generation Failed', fontsize=14, fontweight='bold')
        plt.savefig(os.path.join(results_dir, 'model_comparison_analysis.png'), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

def plot_feature_analysis(df):
    """Generate comprehensive feature analysis histograms"""
    print("[INFO] Generating feature analysis histograms...")
    
    results_dir = get_results_directory()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        print("[WARN] No numeric columns for feature analysis")
        # Create placeholder
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No numeric features available\nfor analysis', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Feature Analysis - No Data', fontsize=14, fontweight='bold')
        plt.savefig(os.path.join(results_dir, 'feature_analysis.png'), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        return
    
    # Create multiple figures for different aspects
    
    # Figure 1: Feature Distributions
    n_features = min(len(numeric_cols), 12)
    fig1, axes1 = plt.subplots(3, 4, figsize=(20, 15))
    axes1 = axes1.flatten()
    
    for i, col in enumerate(numeric_cols[:n_features]):
        if i < len(axes1):
            try:
                sns.histplot(data=df, x=col, ax=axes1[i], kde=True, bins=30, color='skyblue')
                axes1[i].set_title(f'Distribution of {col}', fontsize=10, fontweight='bold')
                axes1[i].tick_params(axis='x', rotation=45)
                
                # Add basic stats
                mean_val = df[col].mean()
                std_val = df[col].std()
                axes1[i].axvline(mean_val, color='red', linestyle='--', alpha=0.7)
                axes1[i].text(0.05, 0.95, f'μ: {mean_val:.2f}\nσ: {std_val:.2f}', 
                             transform=axes1[i].transAxes, va='top', 
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            except Exception as e:
                axes1[i].text(0.5, 0.5, f'Error plotting\n{col}', 
                             ha='center', va='center', transform=axes1[i].transAxes)
                axes1[i].set_title(f'Error: {col}', fontsize=10, fontweight='bold')
    
    # Hide unused subplots
    for i in range(n_features, len(axes1)):
        axes1[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_distributions.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Figure 2: Feature Boxplots
    fig2, axes2 = plt.subplots(3, 4, figsize=(20, 15))
    axes2 = axes2.flatten()
    
    for i, col in enumerate(numeric_cols[:n_features]):
        if i < len(axes2):
            try:
                sns.boxplot(data=df, y=col, ax=axes2[i], color='lightgreen')
                axes2[i].set_title(f'Boxplot of {col}', fontsize=10, fontweight='bold')
                
                # Add outlier info
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                axes2[i].text(0.05, 0.95, f'Outliers: {outliers}', 
                             transform=axes2[i].transAxes, va='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            except Exception as e:
                axes2[i].text(0.5, 0.5, f'Error plotting\n{col}', 
                             ha='center', va='center', transform=axes2[i].transAxes)
                axes2[i].set_title(f'Error: {col}', fontsize=10, fontweight='bold')
    
    for i in range(n_features, len(axes2)):
        axes2[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_boxplots.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Figure 3: Feature Correlation with Target (if target exists)
    target_cols = [col for col in df.columns if 'risk' in col.lower() or 'label' in col.lower() or 'target' in col.lower()]
    if target_cols:
        target_col = target_cols[0]
        if target_col in numeric_cols:
            try:
                # Calculate correlations with target
                correlations = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
                # Remove target itself and get top 10
                top_correlated = correlations.drop(target_col, errors='ignore').head(10)
                
                if len(top_correlated) > 0:
                    fig3, ax3 = plt.subplots(figsize=(12, 8))
                    colors = ['skyblue' if x > 0 else 'lightcoral' for x in top_correlated]
                    top_correlated.plot(kind='barh', ax=ax3, color=colors)
                    ax3.set_xlabel('Absolute Correlation with Target')
                    ax3.set_title('Top Features Correlated with Target', fontsize=14, fontweight='bold')
                    
                    # Add correlation values on bars
                    for i, (feature, corr) in enumerate(top_correlated.items()):
                        ax3.text(corr, i, f' {corr:.3f}', va='center', fontweight='bold')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(results_dir, 'feature_correlations.png'), 
                                dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close()
                else:
                    # Create placeholder for no correlations
                    fig3, ax3 = plt.subplots(figsize=(10, 6))
                    ax3.text(0.5, 0.5, 'No significant correlations\nfound with target', 
                            ha='center', va='center', transform=ax3.transAxes, fontsize=12)
                    ax3.set_title('Feature Correlations - No Significant Results', fontsize=14, fontweight='bold')
                    plt.savefig(os.path.join(results_dir, 'feature_correlations.png'), 
                                dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close()
            except Exception as e:
                print(f"[WARN] Could not create correlation plot: {e}")
                # Create error placeholder
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                ax3.text(0.5, 0.5, f'Error creating correlation plot:\n{str(e)[:50]}...', 
                        ha='center', va='center', transform=ax3.transAxes, fontsize=10)
                ax3.set_title('Feature Correlations - Error', fontsize=14, fontweight='bold')
                plt.savefig(os.path.join(results_dir, 'feature_correlations.png'), 
                            dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
    
    print("[INFO] Feature analysis histograms saved.")

def plot_training_curves():
    """Plot training history and learning curves with informative placeholder"""
    print("[INFO] Generating training curves...")
    
    results_dir = get_results_directory()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    info_text = (
        "Training History Data Not Collected\n\n"
        "Current implementation focuses on final model performance.\n\n"
        "Future Enhancements:\n"
        "• Store epoch-wise metrics for neural networks\n"
        "• Track learning curves for iterative models\n"
        "• Monitor training/validation loss over time\n"
        "• Early stopping visualization\n\n"
        "Current metrics are available in metrics_report.json"
    )
    
    ax.text(0.5, 0.5, info_text, ha='center', va='center', 
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('Training Curves Analysis', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.savefig(os.path.join(results_dir, 'training_curves.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("[INFO] Training curves placeholder saved.")

def generate_all_visualizations(risk_df, original_df=None):
    """Generate all available visualizations in unified directory"""
    print("[INFO] Generating comprehensive visualizations...")
    
    results_dir = get_results_directory()
    print(f"[INFO] All visualizations will be saved to: {results_dir}")
    
    # Generate all types of visualizations
    plot_risk_distribution(risk_df)
    plot_model_comparison()
    
    if original_df is not None:
        plot_feature_analysis(original_df)
    
    plot_training_curves()
    
    print("[INFO] All visualizations generated successfully in unified directory!")