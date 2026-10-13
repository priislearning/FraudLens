import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv("data/creditcard.csv")

X=df.drop(columns=["Class"])
y=df["Class"]

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

print("training features shape:", X_train.shape)
print("testing features shape:", X_test.shape)
print("\n fraud cases in training set:", y_train.sum())
print("fraud cases in testing set:", y_test.sum())