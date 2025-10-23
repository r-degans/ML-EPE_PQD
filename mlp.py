import os
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

def load_dataset(dir):
  frames=[]
  for filename in os.listdir(dir):
    if filename.endswith(".csv"):
      label=os.path.splitext(filename)[0]
      df = pd.read_csv(os.path.join(dir, filename))
      df["Label"] = label
      frames.append(df)
  return pd.concat(frames, ignore_index=True)

file_path = "Files/"
noise_levels = ["SNR_20db", "SNR_30db", "SNR_40db", "SNR_50db", "SNR_noiseless"]

results={}

for snr in noise_levels:
  print(f"SNR value = {snr}:\n")
  data = load_dataset(os.path.join(file_path, snr))
  X = data.iloc[:, :-1]
  y = data["Label"]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

  mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
# in assignment hidden_layer_size=100, alpha=0.0001, shuffle=True    
  )
  mlp.fit(X_train, y_train)
  y_pred = mlp.predict(X_test)

  accuracy = accuracy_score(y_test, y_pred)
  results[snr] = accuracy

  print(f"\nMLP ({snr} db)")
  f1 = f1_score(y_test, y_pred, average='macro')
  print(f"F1-score: {f1:.4f}")
#  print(classification_report(y_test, y_pred))
  print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
  cv_scores = cross_val_score(mlp, X, y, cv=5)
  print("Cross validation accuracy:", cv_scores.mean())

print("MLP Summary")
for snr, accuracy in results.items():
  print(f"{snr}: {accuracy:.4f}")

