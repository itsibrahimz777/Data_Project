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

# Same split as KNN and all other models — fixed seed + stratify for fair comparison
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

# Logistic Regression requires feature scaling (same as KNN, SVM, Neural Network)
lr_scaler = StandardScaler()
features_train_scaled = lr_scaler.fit_transform(features_train)
features_test_scaled = lr_scaler.transform(features_test)

# Hyperparameter tuning via cross-validation on training data only (no leakage)
# C: inverse regularisation strength --> smaller = stronger regularization
# Using lbfgs solver with L2 penalty
# L2 (ridge-style) penalizes large coefficients and is well-suited to this dataset
parameter_grid = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l2"],
    "solver": ["lbfgs"],
    "max_iter": [1000]
}

lr_best_value = GridSearchCV(
    LogisticRegression(random_state=42),
    parameter_grid,
    cv=5,
    scoring="accuracy",
    verbose=2,
    n_jobs=-1
)


# Cross-validation is performed entirely within the training set
lr_best_value.fit(features_train_scaled, labels_train)

lr_model = lr_best_value.best_estimator_

# Evaluation on the held-out test set
accuracy = lr_model.score(features_test_scaled, labels_test)

predicted_labels = lr_model.predict(features_test_scaled)
predicted_label_probabilities = lr_model.predict_proba(features_test_scaled)[:, 1]

lr_confusion_matrix = confusion_matrix(labels_test, predicted_labels)

lr_roc_auc = roc_auc_score(labels_test, predicted_label_probabilities)

# Lower threshold (0.4) to improve recall — catches more true CVD cases
# Important in medical context: missing a CVD case (false negative) is costly
label_predicted_lower = (predicted_label_probabilities > 0.4).astype(int)

# Save the trained model
joblib.dump(lr_model, "lr_model.pkl")

print(f"Best parameters: {lr_best_value.best_params_}")
print(f"Best CV accuracy: {lr_best_value.best_score_:.4f}")

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(labels_test, predicted_labels))

print("\nConfusion Matrix:")
print(lr_confusion_matrix)

print(f"\nROC-AUC: {lr_roc_auc:.4f}")

print("\nWith 0.4 threshold:")
print(classification_report(labels_test, label_predicted_lower))
print(confusion_matrix(labels_test, label_predicted_lower))

# Feature importance via model coefficients
# Logistic Regression gives us a coefficient per feature — shows linear influence
feature_names = features.columns.tolist()
coefficients = lr_model.coef_[0]
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
    "Abs_Coefficient": np.abs(coefficients)
}).sort_values("Abs_Coefficient", ascending=False)

print("\nFeature Importances (by absolute coefficient):")
print(importance_df.to_string(index=False))
