import os
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

print("⚡ Starting ML Pipeline...")

# 1. Generate synthetic dataset
X, y = make_classification(n_samples=200, n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train the Random Forest model cleanly once
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 3. Calculate and save clean, single-backslash text metrics
accuracy = accuracy_score(y_test, y_pred)
with open("metrics.txt", "w") as f:
    f.write(f"Model Accuracy: {accuracy:.4f}\n\n")
    f.write("### Classification Report:\n")
    f.write(classification_report(y_test, y_pred))
print(" Saved metrics.txt")

# 4. Generate and save evaluation plot
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.savefig("confusion_matrix.png", dpi=120)
plt.close()
print(" Saved confusion_matrix.png")
