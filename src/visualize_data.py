import pandas as pd
import matplotlib.pyplot as plt

#load the dataset
df=pd.read_csv("data/creditcard.csv")

#count legitimate and fraudulent transactions
class_counts=df["Class"].value_counts().sort_index()

#create the bar chart
plt.figure(figsize=(7,5))
plt.bar(
    ["Legitimate (0)", "Fraudulent (1)"],
    class_counts.values
)
plt.title("Count of Legitimate vs Fraudulent Transactions")
plt.xlabel("Transaction Class")
plt.ylabel("Number of Transactions")
plt.tight_layout()

#save the chart
plt.show()
plt.close()
