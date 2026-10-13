
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score
)

# Load the dataset
df = pd.read_csv("data/creditcard.csv")

# Separate features and target labels
X = df.drop(columns=["Class"])
y = df["Class"]

# Recreate the same training and testing split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load the trained model
model = joblib.load("results/fraud_model.joblib")

# Predict fraud labels for the test transactions
y_pred = model.predict(X_test)

# Predict fraud probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Display evaluation results
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nPR-AUC (Average Precision):")
print(average_precision_score(y_test, y_prob))
