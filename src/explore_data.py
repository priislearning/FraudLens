
import pandas as pd

# Load the transaction dataset
df = pd.read_csv("data/creditcard.csv")

print("First five transactions:")
print(df.head())

print("\nDataset shape (rows, columns):")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nTransactions in each class:")
print(df["Class"].value_counts())

print("\nFraud percentage:")
print(df["Class"].mean() * 100)
