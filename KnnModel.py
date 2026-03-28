import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

data_frame = pd.read_csv("cardio_clean.csv")

features = data_frame.drop(columns=["cardio"])
labels = data_frame["cardio"]

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

knn_scaler =  StandardScaler()
features_train_scaled = knn_scaler.fit_transform(features_train)
features_test_scaled = knn_scaler.transform(features_test)

parameter_grid = {
    "n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan"]
}

##This finds the best number of neighbors to use:
knn_best_value = GridSearchCV(KNeighborsClassifier(), parameter_grid, cv=5, scoring = "accuracy", verbose = 2)
##Note: cross-validation!!!!
knn_best_value.fit(features_train_scaled, labels_train)

knn_model = knn_best_value.best_estimator_

#To do: Evalutaion (accuracy and stuff)
accuracy = knn_model.score(features_test_scaled, labels_test)

predicted_labels = knn_model.predict(features_test_scaled)
predicted_label_probabilities = knn_model.predict_proba(features_test_scaled)[:, 1]

knn_confusion_matrix = confusion_matrix(labels_test, predicted_labels)

knn_roc_auc = roc_auc_score(labels_test, predicted_label_probabilities)

label_predicted_lower = (predicted_label_probabilities > 0.4).astype(int)

joblib.dump(knn_model, "knn_model.pkl")

print(f"Best parameters: {knn_best_value.best_params_}")
print(f"Best CV accuracy: {knn_best_value.best_score_:.4f}")

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(labels_test, predicted_labels))

print("\nConfusion Matrix:")
print(knn_confusion_matrix)

print(f"\nROC-AUC: {knn_roc_auc:.4f}")

print("\nWith 0.4 threshold:")
print(classification_report(labels_test, label_predicted_lower))
print(confusion_matrix(labels_test, label_predicted_lower))