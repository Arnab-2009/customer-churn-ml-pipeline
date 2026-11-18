"""
End-to-End Telecom Customer Churn Prediction Pipeline
-----------------------------------------------------
Generates a realistic customer dataset, performs preprocessing and feature scaling,
trains and compares Logistic Regression vs. Random Forest classifiers, and exports
evaluation metrics and visualizations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def generate_telecom_data(n_samples: int = 2500, random_state: int = 42) -> pd.DataFrame:
    """Create a realistic synthetic telecom customer dataset."""
    rng = np.random.default_rng(random_state)

    tenure_months = rng.integers(1, 73, size=n_samples)
    contract_type = rng.choice(
        ["Month-to-Month", "One Year", "Two Year"],
        size=n_samples,
        p=[0.55, 0.25, 0.20],
    )
    internet_service = rng.choice(
        ["Fiber Optic", "DSL", "None"],
        size=n_samples,
        p=[0.45, 0.35, 0.20],
    )
    tech_support = rng.choice(["Yes", "No"], size=n_samples, p=[0.35, 0.65])
    support_calls = rng.poisson(lam=1.8, size=n_samples)

    # Monthly charges scale realistically by service tier
    base_charge = np.where(
        internet_service == "Fiber Optic",
        75.0,
        np.where(internet_service == "DSL", 45.0, 20.0),
    )
    monthly_charges = np.round(base_charge + rng.normal(12, 8, size=n_samples), 2)
    monthly_charges = np.clip(monthly_charges, 18.0, 120.0)
    total_charges = np.round(monthly_charges * tenure_months, 2)

    # Logistic churn probability driven by realistic business factors
    logit = (
        -1.2
        - 0.035 * tenure_months
        + 0.025 * (monthly_charges - 50)
        + 0.32 * support_calls
        + np.where(contract_type == "Month-to-Month", 1.15, 0.0)
        + np.where(contract_type == "Two Year", -0.95, 0.0)
        + np.where(tech_support == "No", 0.45, -0.35)
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n_samples) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "support_calls": support_calls,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "tech_support": tech_support,
            "churn": churn,
        }
    )
    return df


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Create a Scikit-Learn preprocessing pipeline for numeric and categorical columns."""
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def save_visualizations(
    y_test: pd.Series,
    y_pred: np.ndarray,
    feature_names: list[str],
    importances: np.ndarray,
    output_dir: str = "outputs",
) -> None:
    """Export clean confusion matrix and feature importance plots for documentation."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Retained (0)", "Churned (1)"],
        yticklabels=["Retained (0)", "Churned (1)"],
    )
    plt.title("Random Forest — Customer Churn Confusion Matrix", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # 2. Feature Importance
    feat_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values(by="Importance", ascending=True)
    )
    plt.figure(figsize=(8, 5))
    plt.barh(feat_df["Feature"], feat_df["Importance"], color="#2563EB")
    plt.title("Top Drivers of Customer Churn (Feature Importance)", fontsize=12, fontweight="bold")
    plt.xlabel("Relative Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=300)
    plt.close()


def main() -> None:
    print("1. Loading dataset...")
    df = generate_telecom_data(n_samples=2500, random_state=42)
    print(f"   Dataset shape: {df.shape} | Churn rate: {df['churn'].mean():.2%}\n")

    X = df.drop(columns=["churn"])
    y = df["churn"]

    numeric_cols = ["tenure_months", "monthly_charges", "total_charges", "support_calls"]
    categorical_cols = ["contract_type", "internet_service", "tech_support"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression (Baseline)": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42
        ),
    }

    best_pipeline = None
    best_auc = 0.0

    print("2. Training and evaluating models...\n")
    for name, classifier in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        print(f"--- {name} ---")
        print(f"Accuracy : {acc:.4f}")
        print(f"ROC-AUC  : {auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

        if auc > best_auc:
            best_auc = auc
            best_pipeline = pipeline

   # Extract feature names and importances (works for both tree models and linear models)
    best_model = best_pipeline.named_steps["classifier"]
    preprocessor = best_pipeline.named_steps["preprocessor"]
    encoded_cat_names = list(
        preprocessor.named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out(categorical_cols)
    )
    all_feature_names = numeric_cols + encoded_cat_names

    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    else:
        importances = np.abs(best_model.coef_[0])

    best_preds = best_pipeline.predict(X_test)
    save_visualizations(
        y_test=y_test,
        y_pred=best_preds,
        feature_names=all_feature_names,
        importances=importances,
    )
    print("3. Saved evaluation plots to ./outputs/ (confusion_matrix.png & feature_importance.png)")


if __name__ == "__main__":
    main()