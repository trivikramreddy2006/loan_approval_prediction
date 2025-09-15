import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv(
    r"DJANGO_COURSE_2.xx\python_learning\loan_prediction_app\data\loan_approval_dataset.csv"
)

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

if "loan_id" in df.columns:
    df = df.drop("loan_id", axis=1)

categorical_cols = ["education", "self_employed", "loan_status"]
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred)
}

print("✅ Model Performance:")
print("Accuracy:", metrics["accuracy"])
print("Precision:", metrics["precision"])
print("Recall:", metrics["recall"])
print("F1 Score:", metrics["f1"])
print("\nClassification Report:\n", classification_report(y_test, y_pred))

with open("loan_model.pkl", "wb") as f:
    pickle.dump(
        {"model": model, "encoders": encoders, "scaler": scaler, "metrics": metrics},
        f
    )
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "coef": model.coef_[0]
}).sort_values(by="coef", ascending=False)
print(feature_importance)
