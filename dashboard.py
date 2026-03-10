import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Page config
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("📊 Telecom Customer Churn Prediction Dashboard")
st.markdown("Identify high-risk customers and explore churn drivers.")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('telco_churn.csv')
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please run generate_data.py first.")
        return None

df = load_data()

if df is not None:
    # Sidebar
    st.sidebar.header("Filters")
    contract_filter = st.sidebar.multiselect("Contract Type", df['Contract'].unique(), default=df['Contract'].unique())
    df_filtered = df[df['Contract'].isin(contract_filter)]

    # Key Metrics
    col1, col2, col3 = st.columns(3)
    churn_rate = (df_filtered['Churn'] == 'Yes').mean()
    col1.metric("Overall Churn Rate", f"{churn_rate:.1%}")
    col2.metric("Total Customers", len(df_filtered))
    col3.metric("Avg Monthly Charges", f"${df_filtered['MonthlyCharges'].mean():.2f}")

    # Row 1: Visualizations
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Churn by Tenure")
        fig, ax = plt.subplots()
        sns.histplot(data=df_filtered, x='tenure', hue='Churn', multiple='stack', ax=ax)
        st.pyplot(fig)

    with col_right:
        st.subheader("Churn by Contract Type")
        fig, ax = plt.subplots()
        sns.countplot(data=df_filtered, x='Contract', hue='Churn', ax=ax)
        st.pyplot(fig)

    # Modeling (Simplified for Dashboard)
    st.markdown("---")
    st.subheader("Model Performance (Logistic Regression)")
    
    # Preprocessing
    X = df.drop(['customerID', 'Churn'], axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numerical_cols),
            ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols)
        ])
    
    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000))])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    col_roc, col_cm = st.columns(2)
    with col_roc:
        st.write(f"**ROC-AUC Score:** {roc_auc:.4f}")
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        st.pyplot(fig)
        
    with col_cm:
        st.write("**Confusion Matrix**")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

    # High Risk Customers
    st.subheader("High Risk Customers (Top 10)")
    df_test = df.loc[X_test.index].copy()
    df_test['Churn Probability'] = y_prob
    high_risk = df_test.sort_values('Churn Probability', ascending=False).head(10)
    st.dataframe(high_risk[['customerID', 'Contract', 'MonthlyCharges', 'tenure', 'Churn Probability']])

