import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import shap
import os

# Set style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def run_pipeline():
    print("--- 1. Data Ingestion ---")
    if not os.path.exists('telco_churn.csv'):
        print("Dataset not found. Please run generate_data.py first.")
        return

    df = pd.read_csv('telco_churn.csv')
    print(f"Data loaded: {df.shape}")

    print("\n--- 2. Exploratory Data Analysis (EDA) ---")
    # Handle missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    print(f"Missing values in TotalCharges: {df['TotalCharges'].isnull().sum()}")
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Class imbalance
    churn_counts = df['Churn'].value_counts()
    print("Class Distribution:\n", churn_counts)
    
    # Visualizations
    plt.figure()
    sns.countplot(x='Churn', data=df)
    plt.title('Churn Distribution')
    plt.savefig('churn_distribution.png')
    print("Saved churn_distribution.png")

    print("\n--- 3. Feature Engineering ---")
    # Tenure buckets
    def tenure_bucket(tenure):
        if tenure < 12: return '0-12'
        elif tenure < 24: return '12-24'
        elif tenure < 48: return '24-48'
        else: return '48+'
    
    df['tenure_group'] = df['tenure'].apply(tenure_bucket)
    
    # Interaction features
    df['MonthlyCharges_Tenure'] = df['MonthlyCharges'] * df['tenure']
    
    # Prepare data for modeling
    X = df.drop(['customerID', 'Churn'], axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    print(f"Categorical features: {list(categorical_cols)}")
    print(f"Numerical features: {list(numerical_cols)}")

    print("\n--- 4. Modeling ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Baseline: Logistic Regression
    lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', LogisticRegression(max_iter=1000))])
    
    print("Training Logistic Regression...")
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)
    y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]

    # Tree-based: Random Forest
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', RandomForestClassifier(random_state=42))])
    
    print("Training Random Forest...")
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
    y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]

    print("\n--- 5. Evaluation ---")
    
    def evaluate_model(name, y_test, y_pred, y_prob):
        print(f"\nResults for {name}:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        auc = roc_auc_score(y_test, y_prob)
        print(f"ROC-AUC Score: {auc:.4f}")
        return auc

    auc_lr = evaluate_model("Logistic Regression", y_test, y_pred_lr, y_prob_lr)
    auc_rf = evaluate_model("Random Forest", y_test, y_pred_rf, y_prob_rf)

    # ROC Curve Plot
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

    plt.figure()
    plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.2f})')
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('roc_curve.png')
    print("Saved roc_curve.png")

    # Feature Importance (Logistic Regression Coefficients)
    # Get feature names after one-hot encoding
    ohe_feature_names = lr_pipeline.named_steps['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out(categorical_cols)
    feature_names = np.r_[numerical_cols, ohe_feature_names]
    
    coefficients = lr_pipeline.named_steps['classifier'].coef_[0]
    
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': coefficients})
    feature_importance['AbsImportance'] = feature_importance['Importance'].abs()
    feature_importance = feature_importance.sort_values(by='AbsImportance', ascending=False).head(10)
    
    plt.figure()
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Top 10 Feature Importance (Logistic Regression)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Saved feature_importance.png")

    print("\n--- Interpretation ---")
    print("Top churn drivers identified from Logistic Regression coefficients:")
    print(feature_importance[['Feature', 'Importance']])

    # SHAP Interpretation
    print("\nGenerating SHAP values (this might take a moment)...")
    try:
        X_test_transformed = preprocessor.transform(X_test)
        if hasattr(X_test_transformed, "toarray"):
            X_test_transformed = X_test_transformed.toarray()
        
        # Check shapes
        model = rf_pipeline.named_steps['classifier']
        print(f"Model expects {model.n_features_in_} features.")
        print(f"Transformed data has {X_test_transformed.shape[1]} features.")
        
        explainer = shap.TreeExplainer(model)
        # check_additivity=False to avoid some numerical precision errors
        shap_values = explainer.shap_values(X_test_transformed, check_additivity=False)
        
        # Summary plot
        plt.figure()
        # For binary classification, shap_values is a list [class0, class1]. We want class1 (Churn=Yes)
        vals = shap_values[1] if isinstance(shap_values, list) else shap_values
        
        shap.summary_plot(vals, X_test_transformed, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig('shap_summary.png')
        print("Saved shap_summary.png")
    except Exception as e:
        print(f"Could not generate SHAP plot: {e}")


if __name__ == "__main__":
    run_pipeline()
