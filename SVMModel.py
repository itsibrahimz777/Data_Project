import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

df = pd.read_csv("cardio_clean.csv")

X = df.drop(columns=["cardio"])
Y = df["cardio"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train_scaled, Y_train)
print("SVM training complete")


model_performance = svm_model.score(X_test_scaled, Y_test)
print(f"Accuracy: {model_performance:.4f}")

Y_pred = svm_model.predict(X_test_scaled)
Y_pred_prob = svm_model.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))

print("Confusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))

roc_auc = roc_auc_score(Y_test, Y_pred_prob)
print(f"\nROC-AUC: {roc_auc:.4f}")