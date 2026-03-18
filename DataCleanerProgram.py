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

#Remove impossible values for blood pressure values

#Threshold we have from our proposal:
# ap_hi (systolic) : keep if 0 < ap_hi < 250
# ap_low (diastolic) : keep if 0 < ap_lo < 150
rows_before = len(df)

df = df[(df["ap_hi"] > 0) & (df["ap_hi"] < 250)]
df = df[(df["ap_lo"] > 0) & (df["ap_lo"] < 150)]

# Also remove rows where diastolic >= systolic (physically impossible)
df = df[df["ap_lo"] < df["ap_hi"]]

rows_after = len(df)
rows_removed = rows_before - rows_after
print(f"Blood pressure outlier removal:")
print(f"  Rows before: {rows_before}")
print(f"  Rows after:  {rows_after}")
print(f"  Removed:     {rows_removed} ({rows_removed / rows_before * 100:.2f}%)\n")

#Quick Sanity check on the cleaned data
print("Cleaned dataset summary:")
print(f"  Shape: {df.shape}")
print(f"  Target distribution:\n{df['cardio'].value_counts().to_string()}")
print(f"\n  ap_hi range: {df['ap_hi'].min()} – {df['ap_hi'].max()}")
print(f"  ap_lo range: {df['ap_lo'].min()} – {df['ap_lo'].max()}")
print(f"  BMI range:   {df['bmi'].min()} – {df['bmi'].max()}")
print(f"  Age range:   {df['age_years'].min()} – {df['age_years'].max()} years")

#Save the cleaned data into cardio_clean.csv
output_path = "cardio_clean.csv"
df.to_csv(output_path, index=False)
print(f"\nCleaned dataset saved to: {output_path}")