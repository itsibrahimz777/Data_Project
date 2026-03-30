import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

data_frame = pd.read_csv("cardio_clean.csv")

features = data_frame.drop(columns=["cardio"])
labels = data_frame["cardio"]