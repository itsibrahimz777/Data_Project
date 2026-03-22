import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow import keras

df = pd.read_csv("cardio_clean.csv")

X = df.drop(["cardio"])
Y = df["cardio"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify = Y)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")