## Decision Tree - run code to get performance metrics
import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


## Load data, combine it
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
  data = load_dataset(os.path.join(file_path, snr))
  X = data.iloc[:, :-1]
  y = data["Label"]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

  ## Model Training
  clf= DecisionTreeClassifier(criterion="entropy", random_state=42)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)

  ## Performance Metrics
  accuracy = accuracy_score(y_test, y_pred)
  results[snr] = accuracy
  print(f"\nDecision Tree (SNR value = {snr})")
  pqd = sorted(y_test.unique())
  f1 = f1_score(y_test, y_pred, average = None, labels = pqd)
  print(f"F1-score:")
  for label, value in zip(pqd, f1):
    print(f"{label}: {value: .4f}")
  f1_average = f1_score(y_test, y_pred, average='macro')
  print(f"F1-score averaged: {f1_average:.4f}")
  print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
  cv_scores = cross_val_score(clf, X, y, cv=5)
  print("Cross validation accuracy:", cv_scores.mean())
