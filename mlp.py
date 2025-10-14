import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_dataset(base_dir):
  frames=[]
  for file in os.listdir(base_dir):
    if file.endswith(".csv"):
      label=os.path.splitext(file)[0]
      df = pd.read_csv(os.path.join(base_dir, file))
      df["Label"] = label
      frames.append(df)
  return pd.concat(frames, ignore_index=True)

base_path = "Files/"
noise_levels = ["SNR_20db", "SNR_30db", "SNR_40db", "SNR_50db", "SNR_noiseless"]

results={}

for snr in noise_levels:
  print(f"SNR {snr}:\n")
  data = load_dataset(os.path.join(base_path, snr))
  X = data.iloc[:, :-1]
  y = data["Label"]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

  mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
  )
  mlp.fit(X_train, y_train)
  y_pred = mlp.predict(X_test)

  acc = accuracy_score(y_test, y_pred)
  results[snr] = acc

  print(f"\nMLP ({snr} db)")
  print(classification_report(y_test, y_pred))
  print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
  cv_scores = cross_val_score(clf, X, y, cv=5)
  print("Cross validation accuracy:", cv_scores.mean())

print("MLP Summary")
for snr, acc in results.items():
  print(f"{snr}: {acc:.4f}")

