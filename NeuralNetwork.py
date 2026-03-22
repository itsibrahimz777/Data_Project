import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


df = pd.read_csv("cardio_clean.csv")


X = df.drop(columns=["cardio"])
Y = df["cardio"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify = Y)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

tf.random.set_seed(42)

model = Sequential()
model.add(InputLayer(input_shape=(12,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
history = model.fit(X_train_scaled, Y_train, epochs=50, batch_size=32, callbacks=[early_stop])

model_performance = model.evaluate(X_test_scaled, Y_test)
print(f"Loss: {model_performance[0]:.4f}")
print(f"Accuracy: {model_performance[1]:.4f}")

Y_pred_prob = model.predict(X_test_scaled)
Y_pred = (Y_pred_prob > 0.5).astype(int).flatten()
Y_pred_lower = (Y_pred_prob > 0.4).astype(int).flatten()

print("\nWith 0.4 threshold:")
print(classification_report(Y_test, Y_pred_lower))
print(confusion_matrix(Y_test, Y_pred_lower))

print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))

print("Confusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))

roc_auc = roc_auc_score(Y_test, Y_pred_prob)
print(f"\nROC-AUC: {roc_auc:.4f}")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()

model.save('neural_network_model.h5')
print("Model saved to neural_network_model.h5")