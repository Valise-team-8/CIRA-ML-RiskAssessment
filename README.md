
# 🔐 CIRA – Cyber Intelligent Risk Assessment for IIoT using Machine Learning

## 📘 Overview

This project implements a **Cyber Intelligent Risk Assessment (CIRA)** system for the **Industrial Internet of Things (IIoT)** using advanced machine learning techniques. The system analyzes cybersecurity datasets, detects threats using STRIDE methodology, and provides comprehensive risk assessments with intelligent scoring and visualization.

## ⚙️ Features

### 🤖 Machine Learning Pipeline
- **Multi-Model Training**: XGBoost, LightGBM, Random Forest classifiers/regressors
- **Federated Learning**: Distributed training simulation across multiple clients
- **Advanced Feature Engineering**: Cybersecurity-specific feature creation
- **Class Imbalance Handling**: SMOTE and class weighting techniques
- **Automated Model Selection**: Performance-based model comparison

### 🛡️ Threat Intelligence
- **STRIDE Threat Modeling**: Spoofing, Tampering, Repudiation, Information Disclosure, DoS, Elevation of Privilege
- **Risk Categorization**: Low, Medium, High, Critical threat levels
- **Vulnerability Assessment**: Comprehensive risk scoring (0-1 scale)
- **Security Controls**: Risk-level specific mitigation strategies

### 📊 Advanced Analytics & Visualization
- **Comprehensive Risk Analysis**: Multi-dimensional risk distribution charts
- **Model Performance Comparison**: Detailed metrics visualization
- **Feature Analysis**: Distribution plots, correlation heatmaps, boxplots
- **Interactive Dashboards**: Risk trends and threat level monitoring

### 🔧 Data Processing
- **Multi-Dataset Integration**: Automatic CSV loading and merging
- **Data Validation**: Comprehensive quality checks and preprocessing
- **Missing Value Handling**: Intelligent imputation strategies

## 🧠 Tech Stack

**Core Technologies:**
- **Python 3.11+** - Primary development language
- **Scikit-learn** - Machine learning algorithms
- **XGBoost & LightGBM** - Gradient boosting frameworks
- **Pandas & NumPy** - Data manipulation and analysis
- **TensorFlow** - Deep learning capabilities

**Visualization & Analysis:**
- **Matplotlib & Seaborn** - Statistical visualizations
- **Plotly** - Interactive charts

**Specialized Libraries:**
- **Imbalanced-learn** - Class imbalance handling
- **Flower (flwr)** - Federated learning framework
- **Joblib** - Model serialization

## 📁 Project Structure
CIRA-ML-RiskAssessment/
├── 📂 data/
│ ├── cybersecurity_risks.csv
│ ├── cybersecurity_attacks.csv
│ ├── CloudWatch_Traffic_Web_Attack.csv
│ └── ... (additional datasets)
├── 📂 src/
│ ├── main.py
│ ├── data_preprocessing.py
│ ├── centralized_training.py
│ ├── federated_training.py
│ ├── risk_ranking.py
│ ├── threat_modeling_stride.py
│ ├── visualization.py
│ └── security_controls.py
├── 📂 models/
│ ├── xgboost_model.pkl
│ ├── lightgbm_model.pkl
│ └── random_forest_model.pkl
└── 📂 results/
├── comprehensive_risk_analysis.png
├── model_comparison_analysis.png
├── feature_distributions.png
├── metrics_report.json
├── risk_summary.json
└── risk_table.csv

text

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Valise-team-8/CIRA-ML-RiskAssessment.git
cd CIRA-ML-RiskAssessment

# Install dependencies
pip install -r requirements.txt

# Run the complete CIRA pipeline
python src/main.py
Alternative: Conda Environment
bash
# Create conda environment
conda create -n cira-env python=3.11
conda activate cira-env

# Install dependencies
pip install -r requirements.txt

# Run pipeline
python src/main.py
📊 Pipeline Workflow
1. Data Loading & Preprocessing
python
# Loads multiple cybersecurity datasets
# Performs feature engineering and validation
# Handles data cleaning and preparation
2. STRIDE Threat Modeling
python
# Applies STRIDE methodology
# Generates threat profiles and vulnerability types
# Maps threats to security categories
3. Machine Learning Training
python
# Centralized training: XGBoost, LightGBM, Random Forest
# Federated learning simulation
# Automated model training and evaluation
4. Risk Assessment
python
# Calculates comprehensive risk scores (0-1)
# Categorizes risks: Low, Medium, High, Critical
# Generates risk analysis and reporting
5. Visualization & Reporting
python
# Creates comprehensive chart types
# Generates analysis reports
# Saves results to output directory
📈 Key Outputs
Generated Visualizations
Risk Distribution Analysis: Pie charts, histograms, boxplots

Model Performance Comparison: Accuracy, F1-score, ROC-AUC metrics

Feature Analysis: Correlation heatmaps, distribution plots

Threat Level Monitoring: Category distributions and trends

Data Exports
risk_table.csv: Complete risk assessment results

metrics_report.json: Model performance metrics

risk_summary.json: Statistical risk analysis

security_controls.json: Recommended mitigation strategies

Model Artifacts
Trained Models: Serialized ML models for deployment

Feature Mappings: Column transformations and encodings

Performance Metrics: Cross-validation and test results

📋 Requirements
Core Dependencies
text
pandas==2.1.0          # Data manipulation
numpy==1.26.4          # Numerical computing
scikit-learn==1.3.2    # Machine learning
xgboost==2.0.3         # Gradient boosting
lightgbm==4.3.0        # Light gradient boosting
matplotlib==3.8.4      # Plotting
seaborn==0.13.2        # Statistical visualization
plotly==5.22.0         # Interactive charts
joblib==1.4.2          # Model serialization
flwr==1.7.0            # Federated learning
tensorflow-cpu==2.14.0 # Deep learning
imbalanced-learn==0.12.2 # Class imbalance handling
🎯 Use Cases
Enterprise Security
SOC Integration: Real-time threat assessment

Risk Management: Quantitative security metrics

Compliance Reporting: Automated risk documentation

Research & Development
Algorithm Comparison: Multi-model performance analysis

Dataset Analysis: Cybersecurity data exploration

Threat Intelligence: STRIDE-based vulnerability assessment

Educational Applications
Cybersecurity Training: Hands-on ML security projects

Risk Assessment Learning: Practical threat modeling

Data Science Education: Real-world security datasets





<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/4505fe1b-5e9e-4970-a6ac-26ae315f18c5" />

