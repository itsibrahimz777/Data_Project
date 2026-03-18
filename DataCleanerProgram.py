import pandas as pd
import numpy as np


#Loading the raw data

df = pd.read_csav("cardio_train.csv", sep = ";")

print(f"Raw dataset shape: {df.shape}")
print(f"Raw dataset columns: {df.columns}\n")

#We do not need id
df = df.drop(columns=["id"])

#Check for missing values
null_counts = df.isnull().sum()
print("Missing values per column:")
print(null_counts)
print()

if null_counts.sum() == 0:
    print("No missing values found - no imputation needed.\n")
else:
    print("Warning: Missing values detected - imputation would be needed.\n")



