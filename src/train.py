import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df=pd.read_csv("data/creditcard.csv")

X=df.drop(columns=["Class"])
y=df["Class"]

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
 
model=Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

model.fit(X_train,y_train)

joblib.dump(model,"results/fraud_model.joblib")

print("Model training complete")
print("Training transactions:",len(X_train))
print("Testing transactions:",len(X_test))
print("trained model saved to results")
