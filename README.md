# Customer Churn Prediction in Music Streaming Services

## Overview

This project implements a comprehensive machine learning pipeline to predict customer churn for a music streaming service. The solution leverages multiple data sources including user demographics, transaction history, and listening behavior to build an ensemble model that predicts whether a customer will churn.

## Problem Statement

Customer churn prediction is critical for subscription-based businesses like music streaming services. By identifying customers likely to cancel their subscriptions, companies can proactively implement retention strategies, ultimately reducing revenue loss and improving customer lifetime value.

## Dataset Description

The project uses five main datasets:

1. **train_data.csv** - Training labels with user IDs and churn status
2. **members.csv** - User demographic information (age, gender, city, registration details)
3. **transactions.csv** - Payment and subscription transaction history
4. **user_logs.csv** - Daily listening behavior and engagement metrics
5. **kaggle_test_data.csv** - Test set for final predictions

## Features Engineering

### Member Features
- **Demographics**: Age groups, gender, city, registration channel
- **Temporal**: Registration year/month, days since registration
- **Derived**: Age grouping and registration tenure

### Transaction Features
- **Payment patterns**: Payment method diversity, plan preferences
- **Financial metrics**: Average amounts, discount rates, price variations
- **Subscription behavior**: Auto-renewal patterns, cancellation history
- **Timing**: Days until membership expiration

### Behavioral Features
- **Listening patterns**: Song completion rates, skip rates, daily activity
- **Engagement metrics**: Total listening time, unique songs played
- **Usage intensity**: Average daily songs and listening duration
- **Activity consistency**: Number of active days, usage patterns

## Model Architecture

### Individual Models
- **Random Forest**: Robust ensemble method with 200 trees
- **Gradient Boosting**: Sequential boosting for pattern learning
- **Logistic Regression**: Linear baseline model
- **LightGBM**: Efficient gradient boosting framework
- **XGBoost**: Advanced gradient boosting implementation

### Ensemble Strategy
- **Weighted averaging**: Models weighted by validation AUC scores
- **Performance-based weighting**: Higher performing models get more influence
- **Robust predictions**: Reduces individual model bias and variance

## Key Technical Features

### Data Preprocessing
- **Missing value handling**: Median imputation for numerical features
- **Categorical encoding**: Label encoding for categorical variables
- **Date processing**: Conversion to datetime and feature extraction
- **Data validation**: Comprehensive checks for data quality

### Feature Engineering Pipeline
- **Multi-source integration**: Combines demographic, transactional, and behavioral data
- **Derived metrics**: Creates meaningful ratios and aggregations
- **Temporal features**: Captures time-based patterns and trends

### Model Training & Evaluation
- **Cross-validation**: Stratified train-validation split
- **Performance metrics**: AUC-ROC for binary classification
- **Feature importance**: Analysis of most predictive features
- **Ensemble optimization**: Weighted combination based on individual performance

## Installation & Setup

### Prerequisites
```bash
pip install pandas numpy scikit-learn lightgbm xgboost matplotlib seaborn
```

### Required Libraries
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, lightgbm, xgboost
- **Visualization**: matplotlib, seaborn
- **Utilities**: warnings, datetime

## Usage

### Basic Usage
```python
# Initialize the model
model = ChurnPredictionModel()

# Run complete pipeline
submission = model.run_complete_pipeline()

# Generate feature importance plot
model.plot_feature_importance()
```

### Pipeline Steps
1. **Data Loading**: Load all CSV files
2. **Preprocessing**: Clean and validate data
3. **Feature Engineering**: Create features from all data sources
4. **Model Training**: Train multiple algorithms
5. **Ensemble Creation**: Combine models with optimal weights
6. **Prediction Generation**: Create final predictions
7. **Output**: Generate submission file

## Model Performance

### Evaluation Metrics
- **Primary Metric**: AUC-ROC (Area Under Curve)
- **Individual Models**: Each model evaluated separately
- **Ensemble Performance**: Weighted combination typically outperforms individual models

### Feature Importance
The model automatically generates feature importance plots to identify:
- Most predictive user characteristics
- Key behavioral indicators of churn
- Important transaction patterns

## Output

### Submission File
- **Format**: CSV with columns `msno` (user ID) and `is_churn` (probability)
- **File**: `submission.csv`
- **Usage**: Ready for Kaggle competition submission

## Project Structure

```
├── customer-churn-prediction-in-music-streaming-servi.ipynb
├── README.md
├── submission.csv (generated)
└── data/
    ├── train_data.csv
    ├── members.csv
    ├── transactions.csv
    ├── user_logs.csv
    └── kaggle_test_data.csv
```

## Key Insights

### Critical Churn Indicators
- **Behavioral patterns**: Listening frequency and engagement levels
- **Transaction history**: Payment patterns and subscription changes
- **User demographics**: Age groups and registration channels
- **Temporal factors**: Time since registration and activity recency

### Business Applications
- **Targeted retention**: Focus on high-risk customer segments
- **Personalized offers**: Tailor retention strategies based on user profiles
- **Product optimization**: Improve features that drive engagement
- **Revenue protection**: Proactive intervention before churn occurs

## Advanced Features

### Robust Error Handling
- **Data validation**: Comprehensive checks for data integrity
- **Missing category handling**: Graceful handling of unseen categorical values
- **Type conversion**: Automatic conversion to appropriate data types

### Scalable Architecture
- **Modular design**: Separate methods for each pipeline component
- **Extensible framework**: Easy to add new features or models
- **Memory efficient**: Optimized for large datasets

## Future Enhancements

### Potential Improvements
- **Deep learning models**: Neural networks for complex pattern recognition
- **Time series features**: Incorporate temporal trends and seasonality
- **Advanced ensembling**: Stacking and blending techniques
- **Feature selection**: Automated feature importance and selection
- **Hyperparameter optimization**: Grid search and Bayesian optimization

### Model Monitoring
- **Performance tracking**: Monitor model performance over time
- **Data drift detection**: Identify changes in data distribution
- **Model retraining**: Automated pipeline for model updates

## Contributing

### Guidelines
1. Follow existing code structure and naming conventions
2. Add comprehensive documentation for new features
3. Include unit tests for critical functions
4. Update README for significant changes

### Development Setup
1. Clone the repository
2. Install required dependencies
3. Set up data directory with required CSV files
4. Run the notebook to validate setup

## License

This project is created for educational and competition purposes. Please ensure appropriate licensing for commercial use.

## Acknowledgments

- **Data Source**: Customer Retention Datathon APAC Edition
- **Libraries**: Thanks to the open-source community for excellent ML libraries
- **Competition**: Kaggle platform for hosting the challenge

---

**Note**: This model is designed for the specific dataset structure and may require modifications for different data formats or business contexts.
