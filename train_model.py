import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------- Load dataset ----------------
df = pd.read_csv("data/loan_data.csv")

# Clean column names & string values
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# ---------------- Encode categorical INPUT columns ----------------
le = LabelEncoder()
categorical_cols = ["education", "self_employed"]

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# ---------------- Encode TARGET (BEST PRACTICE) ----------------
df["loan_status"] = df["loan_status"].map({
    "Approved": 1,
    "Rejected": 0
})

# ---------------- Features & Target ----------------
X = df.drop(["loan_status", "loan_id"], axis=1)
y = df["loan_status"]

# ✅ Save feature names for Streamlit
feature_names = X.columns.tolist()
pickle.dump(feature_names, open("model/features.pkl", "wb"))

# ---------------- Train-test split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- Feature Scaling ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- Logistic Regression ----------------
log_model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    solver="lbfgs"
)
log_model.fit(X_train_scaled, y_train)

# ---------------- Improved Random Forest ----------------
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)

# ---------------- Evaluation ----------------
log_acc = accuracy_score(y_test, log_model.predict(X_test_scaled))
rf_acc = accuracy_score(y_test, rf_model.predict(X_test_scaled))

print(f"✅ Logistic Regression Accuracy: {log_acc * 100:.2f}%")
print(f"✅ Random Forest Accuracy: {rf_acc * 100:.2f}%")

# ---------------- Cross Validation (Interview Gold) ----------------
cv_scores = cross_val_score(
    rf_model,
    X_train_scaled,
    y_train,
    cv=5
)

print(f"✅ Random Forest CV Accuracy: {cv_scores.mean() * 100:.2f}%")

# ---------------- Save models ----------------
pickle.dump(log_model, open("model/logistic_model.pkl", "wb"))
pickle.dump(rf_model, open("model/rf_model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))

print("✅ Models trained and saved successfully")
print("✅ Features used:", feature_names)
