# rf_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
# -------------------------------
# Step 1. Load weekly dataset
# -------------------------------
df = pd.read_csv("weekly_prepared.csv")

print("✅ Weekly dataset loaded")
print(df.head())

# -------------------------------
# Step 2. Features (X) and Target (y)
# -------------------------------
X = df.drop(columns=["Date", "Outbreak"])
y = df["Outbreak"]

# Handle missing values
X = X.fillna(0)

# -------------------------------
# Step 3. Train/Test Split
# -------------------------------
if y.nunique() > 1 and y.value_counts().min() >= 2:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
else:
    print("⚠️ Warning: Imbalanced outbreak data, using random split without stratify")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

# -------------------------------
# Step 4. Random Forest Training
# -------------------------------
rf = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train, y_train)

# -------------------------------
# Step 5. Predictions + Evaluation
# -------------------------------
y_pred = rf.predict(X_test)

print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Outbreak", "Outbreak"], yticklabels=["No Outbreak", "Outbreak"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()

# -------------------------------
# Step 6. Feature Importance
# -------------------------------
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(10,5))
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45, ha="right")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()

joblib.dump(rf, "rf_dengue_model.pkl")
print("💾 Random Forest model saved as rf_dengue_model.pkl")