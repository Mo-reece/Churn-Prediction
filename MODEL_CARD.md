# Model Card: Customer Churn Prediction

## Project Overview

This project predicts whether a customer is likely to churn and explains the business drivers behind that prediction. It combines exploratory analysis, feature engineering, classification models, ROC-AUC evaluation, and visual outputs suitable for a retention-focused stakeholder discussion.

## Intended Use

The model is intended for portfolio demonstration and business analytics practice. It can help frame how a telecom or SaaS team might identify higher-risk customer segments and prioritize retention outreach.

## Data

The repository uses a synthetic Telco-style churn dataset generated for demonstration. Synthetic data is useful for reproducibility and privacy, but it does not prove production performance on real customer behavior.

## Evaluation

The project evaluates model performance using classification metrics, confusion matrices, ROC-AUC, feature importance, and SHAP-style explainability outputs.

## Limitations

- Synthetic data limits real-world validity.
- No production monitoring, drift detection, or retraining pipeline is included.
- Business actions should be validated with real customer outcomes and A/B testing before deployment.

## Future Improvements

- Add a saved model artifact and reproducible prediction CLI.
- Add cross-validation reports and calibration analysis.
- Deploy the dashboard as a Streamlit app.
- Add tests for data generation and feature engineering.
