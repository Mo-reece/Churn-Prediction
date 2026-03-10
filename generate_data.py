import pandas as pd
import numpy as np

def generate_churn_data(n_samples=7043):
    np.random.seed(42)
    
    # Customer ID
    customer_ids = [f'{np.random.randint(1000,9999)}-{np.random.choice(["A","B","C","D"])}{np.random.randint(100,999)}' for _ in range(n_samples)]
    
    # Demographics
    gender = np.random.choice(['Male', 'Female'], n_samples)
    senior_citizen = np.random.choice([0, 1], n_samples, p=[0.84, 0.16])
    partner = np.random.choice(['Yes', 'No'], n_samples)
    dependents = np.random.choice(['Yes', 'No'], n_samples)
    
    # Services
    tenure = np.random.randint(0, 73, n_samples)
    phone_service = np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1])
    multiple_lines = np.where(phone_service == 'Yes', np.random.choice(['Yes', 'No', 'No phone service'], n_samples), 'No phone service')
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.34, 0.44, 0.22])
    
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    service_cols = {}
    for service in services:
        service_cols[service] = np.where(internet_service != 'No', 
                                         np.random.choice(['Yes', 'No'], n_samples), 
                                         'No internet service')
    
    # Account info
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.21, 0.24])
    paperless_billing = np.random.choice(['Yes', 'No'], n_samples)
    payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples)
    
    # Charges
    monthly_charges = np.random.uniform(18.25, 118.75, n_samples)
    # Correlate total charges with tenure and monthly charges
    total_charges = monthly_charges * tenure + np.random.normal(0, 10, n_samples)
    total_charges = np.where(total_charges < 0, 0, total_charges) # No negative charges
    
    # Target: Churn
    # Make churn correlated with some features to make the model work
    # Higher churn for Month-to-month, Fiber optic, Electronic check
    churn_prob = 0.15
    churn_prob += np.where(contract == 'Month-to-month', 0.3, 0.0)
    churn_prob += np.where(internet_service == 'Fiber optic', 0.15, 0.0)
    churn_prob += np.where(payment_method == 'Electronic check', 0.1, 0.0)
    churn_prob -= np.where(tenure > 24, 0.2, 0.0)
    churn_prob -= np.where(total_charges > 2000, 0.1, 0.0)
    
    churn_prob = np.clip(churn_prob, 0, 1)
    churn = [np.random.choice(['Yes', 'No'], p=[p, 1-p]) for p in churn_prob]
    
    data = pd.DataFrame({
        'customerID': customer_ids,
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        **service_cols,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Churn': churn
    })
    
    # Introduce some missing values in TotalCharges as in original dataset
    mask = np.random.random(n_samples) < 0.005
    data.loc[mask, 'TotalCharges'] = np.nan
    
    return data

if __name__ == "__main__":
    print("Generating synthetic Telco Churn dataset...")
    df = generate_churn_data()
    df.to_csv('telco_churn.csv', index=False)
    print("Dataset saved to telco_churn.csv")
