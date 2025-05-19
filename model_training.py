import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Sample synthetic data
data = {
    'age': [25, 45, 35, 33, 50, 23, 31, 40],
    'income': [50000, 100000, 60000, 65000, 120000, 40000, 58000, 90000],
    'loan_amount': [2000, 10000, 3000, 5000, 15000, 1800, 2500, 8500],
    'credit_risk': [0, 0, 0, 0, 1, 1, 0, 1]  # 0 = low risk, 1 = high risk
}

df = pd.DataFrame(data)

X = df[['age', 'income', 'loan_amount']]
y = df['credit_risk']

model = LogisticRegression()
model.fit(X, y)

# Save the model to a file
joblib.dump(model, 'credit_risk_model.pkl')
print("✅ Model saved as 'credit_risk_model.pkl'")
