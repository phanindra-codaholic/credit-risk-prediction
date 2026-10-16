import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "credit_risk_dataset.csv")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "notebooks", "figures")

def run_eda():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    if not os.path.exists(DATA_PATH):
        from data.download_data import download_dataset
        download_dataset()

    df = pd.read_csv(DATA_PATH)
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS (EDA) REPORT")
    print("=" * 60)
    print(f"Shape of Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\n--- Columns and Data Types ---")
    print(df.dtypes)
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")
    print("\n--- Target Variable Distribution (`loan_status`) ---")
    counts = df["loan_status"].value_counts()
    percentages = df["loan_status"].value_counts(normalize=True) * 100
    for val in counts.index:
        label = "Default (1) [High Risk]" if val == 1 else "Non-Default (0) [Low Risk]"
        print(f"  {label}: {counts[val]} rows ({percentages[val]:.2f}%)")

    # 1. Target Distribution Plot
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="loan_status", palette="coolwarm")
    plt.title("Loan Status Distribution (0 = Non-Default, 1 = Default)")
    plt.xlabel("Loan Status")
    plt.ylabel("Count")
    plt.tight_layout()
    target_plot = os.path.join(FIGURES_DIR, "target_distribution.png")
    plt.savefig(target_plot)
    plt.close()
    print(f"\nSaved target distribution figure to: {target_plot}")

    # 2. Outliers in Age and Employment Length
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(data=df, x="person_age", ax=axes[0], color="skyblue")
    axes[0].set_title("Distribution of Age (Notice impossible > 100 values)")
    sns.boxplot(data=df, x="person_emp_length", ax=axes[1], color="salmon")
    axes[1].set_title("Distribution of Employment Length (Notice impossible > 100 values)")
    plt.tight_layout()
    outlier_plot = os.path.join(FIGURES_DIR, "outliers_age_emp.png")
    plt.savefig(outlier_plot)
    plt.close()
    print(f"Saved outlier analysis figure to: {outlier_plot}")

    # 3. Loan Intent vs Target
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x="loan_intent", hue="loan_status", palette="Set2")
    plt.title("Loan Intent by Loan Status")
    plt.xticks(rotation=30)
    plt.tight_layout()
    intent_plot = os.path.join(FIGURES_DIR, "loan_intent_vs_target.png")
    plt.savefig(intent_plot)
    plt.close()
    print(f"Saved loan intent relationship figure to: {intent_plot}")

    # 4. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Numerical Features Correlation Matrix")
    plt.tight_layout()
    corr_plot = os.path.join(FIGURES_DIR, "correlation_heatmap.png")
    plt.savefig(corr_plot)
    plt.close()
    print(f"Saved correlation heatmap to: {corr_plot}")

    print("\nEDA completed successfully.")

if __name__ == "__main__":
    run_eda()
