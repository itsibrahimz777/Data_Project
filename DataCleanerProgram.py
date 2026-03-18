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

#Change Age from days -> Years, add BMI calculator

#convert age from days to years rounding to one decimal
df["age_years"] = np.round(df["age"]/ 365.25, 1)

#get BMI from formula : BMI = weight(kg) / height(m)^2
#Height in cm so we first convert to meters
df["bmi"] = np.round(df["weight"]/ (df["height"]/100) **2,2)

#drop the original "age" column, since we have age_years now
df = df.drop(columns=["age"])

print(f"Age range(years): {df['age_years'].min()} - {df['age_years'].max()}")
print(f"BMI range: {df['bmi'].min()} - {df['bmi'].max()}\n")



