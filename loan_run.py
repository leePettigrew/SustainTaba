import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# ----------------------------
# 1) Load data
# ----------------------------
df = pd.read_csv("train.csv")

# Basic sanity checks
print("Shape:", df.shape)
print("Columns:", list(df.columns))

# ----------------------------
# 2) Split features/target
# ----------------------------
y = (df["Loan_Status"] == "Y").astype(int)
X = df.drop(columns=["Loan_Status", "Loan_ID"])

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

print("\nCategorical cols:", cat_cols)
print("Numeric cols:", num_cols)

# ----------------------------
# 3) Train/test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ----------------------------
# 4) Preprocessing pipeline
# ----------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ]
)

# ----------------------------
# 5) Models
# ----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, solver="liblinear"),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

results = []
preds_by_model = {}
probas_by_model = {}
train_time_by_model = {}

# ----------------------------
# 6) Train + evaluate
# ----------------------------
for name, model in models.items():
    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])

    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    t1 = time.perf_counter()

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    preds_by_model[name] = y_pred
    probas_by_model[name] = y_proba
    train_time_by_model[name] = t1 - t0

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_proba),
        "Train_time_s": t1 - t0
    })

metrics_df = pd.DataFrame(results).sort_values("F1", ascending=False)
print("\n=== Model Metrics (Test Set) ===")
print(metrics_df)

# Save metrics table
metrics_df.to_csv("model_metrics.csv", index=False)

# ----------------------------
# 7) Fairness analysis (Gender + Education)
# ----------------------------
def fairness_table(attr: str) -> pd.DataFrame:
    rows = []
    base = X_test[[attr]].copy()
    base["y_true"] = y_test.values

    for model_name in models.keys():
        tmp = base.copy()
        tmp["y_pred"] = preds_by_model[model_name]

        for grp, gdf in tmp.groupby(attr, dropna=False):
            pred_rate = gdf["y_pred"].mean()
            true_rate = gdf["y_true"].mean()

            # True Positive Rate (Equal Opportunity proxy)
            positives = gdf[gdf["y_true"] == 1]
            tpr = positives["y_pred"].mean() if len(positives) > 0 else np.nan

            rows.append({
                "Model": model_name,
                attr: str(grp),
                "n": len(gdf),
                "True_approval_rate": true_rate,
                "Predicted_approval_rate": pred_rate,
                "TPR": tpr
            })

    return pd.DataFrame(rows)

gender_df = fairness_table("Gender")
edu_df = fairness_table("Education")

gender_df.to_csv("fairness_gender.csv", index=False)
edu_df.to_csv("fairness_education.csv", index=False)

print("\n=== Fairness (Gender) ===")
print(gender_df)
print("\n=== Fairness (Education) ===")
print(edu_df)

# ----------------------------
# 8) Graphs (required)
# ----------------------------
# Graph 1: F1 by model
plt.figure()
plt.bar(metrics_df["Model"], metrics_df["F1"])
plt.ylabel("F1 Score")
plt.title("F1-score Comparison Across Models")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("fig_f1_by_model.png")
plt.close()

# Graph 2: Accuracy vs training time
plt.figure()
plt.scatter(metrics_df["Train_time_s"], metrics_df["Accuracy"])
for i in range(len(metrics_df)):
    plt.text(metrics_df["Train_time_s"].iloc[i], metrics_df["Accuracy"].iloc[i], metrics_df["Model"].iloc[i])
plt.xlabel("Training Time (s) [Efficiency proxy]")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Efficiency Trade-off")
plt.tight_layout()
plt.savefig("fig_accuracy_vs_time.png")
plt.close()

# Graph 3: Predicted approval rate by Gender to have a talking point on fairness
plt.figure()
for model_name in models.keys():
    tmp = X_test[["Gender"]].copy()
    tmp["y_pred"] = preds_by_model[model_name]
    grp_rates = tmp.groupby("Gender")["y_pred"].mean()
    plt.plot(grp_rates.index.astype(str), grp_rates.values, marker="o", label=model_name)
plt.ylabel("Predicted Approval Rate")
plt.title("Predicted Approval Rate by Gender")
plt.legend()
plt.tight_layout()
plt.savefig("fig_pred_approval_by_gender.png")
plt.close()

print("\nSaved outputs:")
print("- model_metrics.csv")
print("- fairness_gender.csv")
print("- fairness_education.csv")
print("- fig_f1_by_model.png")
print("- fig_accuracy_vs_time.png")
print("- fig_pred_approval_by_gender.png")
