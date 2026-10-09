import sys
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    confusion_matrix
)
import warnings

warnings.filterwarnings("ignore")


# CHANGED: Validate the dataset and generate synthetic data if it is unavailable.
def load_or_generate_data(
    file_path="../data/creditcard.csv",
    n_samples=100000,
    random_state=42
):
    try:
        df = pd.read_csv(file_path)
        print(f"[+] Loaded real dataset: {df.shape[0]} transactions.")

    except FileNotFoundError:
        print(
            f"[-] '{file_path}' not found. "
            f"Generating synthetic benchmark dataset ({n_samples} rows)..."
        )

        rng = np.random.default_rng(random_state)

        n_fraud = max(1, int(n_samples * 0.0017))
        n_normal = n_samples - n_fraud

        normal_v = rng.normal(0.0, 1.0, size=(n_normal, 28))
        normal_amt = rng.exponential(85.0, size=(n_normal, 1))
        normal_time = rng.uniform(0, 172800, size=(n_normal, 1))
        normal_labels = np.zeros((n_normal, 1))

        fraud_v = rng.normal(0.0, 1.0, size=(n_fraud, 28))
        fraud_v[:, 0] -= 2.5
        fraud_v[:, 2] += 3.0
        fraud_v[:, 11] -= 3.5

        fraud_amt = rng.exponential(180.0, size=(n_fraud, 1))
        fraud_time = rng.uniform(0, 172800, size=(n_fraud, 1))
        fraud_labels = np.ones((n_fraud, 1))

        normal_data = np.hstack([
            normal_time,
            normal_v,
            normal_amt,
            normal_labels
        ])

        fraud_data = np.hstack([
            fraud_time,
            fraud_v,
            fraud_amt,
            fraud_labels
        ])

        all_data = np.vstack([normal_data, fraud_data])

        col_names = (
            ["Time"]
            + [f"V{i}" for i in range(1, 29)]
            + ["Amount", "Class"]
        )

        df = pd.DataFrame(all_data, columns=col_names)
        df = df.sort_values(by="Time").reset_index(drop=True)
        df["Class"] = df["Class"].astype(int)

    # ADDED: Validate required columns for both real and synthetic datasets.
    required_columns = (
        ["Time", "Amount", "Class"]
        + [f"V{i}" for i in range(1, 29)]
    )

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("The dataset is empty.")

    # ADDED: Ensure the target contains both classes.
    if df["Class"].isna().any():
        raise ValueError("The Class column contains missing values.")

    if not set(df["Class"].unique()).issubset({0, 1}):
        raise ValueError("The Class column must contain only 0 and 1.")

    if df["Class"].nunique() != 2:
        raise ValueError(
            "The dataset must contain both normal (0) and fraud (1) "
            "transactions."
        )

    # ADDED: Reject missing or infinite numeric values.
    numeric_columns = required_columns

    if df[numeric_columns].isna().any().any():
        raise ValueError("Dataset contains missing numeric values.")

    if not np.isfinite(df[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains infinite numeric values.")

    return df


# CHANGED: Fit scalers on training data only to prevent data leakage.
def preprocess_features(df):
    df_clean = df.copy()

    feature_cols = (
        [f"V{i}" for i in range(1, 29)]
        + ["scaled_amount", "scaled_time"]
    )

    # ADDED: Validate data before preprocessing.
    if "Class" not in df_clean.columns:
        raise ValueError("Dataset must contain a Class column.")

    amount_scaler = RobustScaler()
    time_scaler = RobustScaler()

    # Return unscaled feature values for splitting.
    df_clean["scaled_amount"] = df_clean["Amount"]
    df_clean["scaled_time"] = df_clean["Time"]

    df_clean = df_clean.drop(["Time", "Amount"], axis=1)

    X = df_clean[feature_cols]
    y = df_clean["Class"].astype(int)

    return X, y, feature_cols, amount_scaler, time_scaler


# ADDED: Apply scalers using training-set statistics only.
def scale_features(X_train, X_val, amount_scaler, time_scaler):
    X_train = X_train.copy()
    X_val = X_val.copy()

    X_train["scaled_amount"] = amount_scaler.fit_transform(
        X_train[["scaled_amount"]]
    ).ravel()

    X_val["scaled_amount"] = amount_scaler.transform(
        X_val[["scaled_amount"]]
    ).ravel()

    X_train["scaled_time"] = time_scaler.fit_transform(
        X_train[["scaled_time"]]
    ).ravel()

    X_val["scaled_time"] = time_scaler.transform(
        X_val[["scaled_time"]]
    ).ravel()

    return X_train, X_val


# CHANGED: Evaluate financial cost while handling confusion-matrix edge cases.
def evaluate_financial_cost(y_true, y_probs, cost_fn=150.0, cost_fp=10.0):
    thresholds = np.linspace(0.01, 0.99, 100)

    best_threshold = 0.5
    min_cost = float("inf")
    metrics_at_best = {}

    for threshold in thresholds:
        preds = (y_probs >= threshold).astype(int)

        # ADDED: Specify labels to guarantee a 2x2 confusion matrix.
        tn, fp, fn, tp = confusion_matrix(
            y_true,
            preds,
            labels=[0, 1]
        ).ravel()

        total_cost = (fn * cost_fn) + (fp * cost_fp)

        if total_cost < min_cost:
            min_cost = total_cost
            best_threshold = threshold

            metrics_at_best = {
                "TP": int(tp),
                "FP": int(fp),
                "TN": int(tn),
                "FN": int(fn),
                "Cost": float(total_cost)
            }

    return best_threshold, min_cost, metrics_at_best


# CHANGED: Validate columns and sample sizes before running drift detection.
class DataDriftMonitor:
    def __init__(
        self,
        baseline_df,
        alpha=0.01,
        threshold_flagged_ratio=0.15
    ):
        self.baseline_df = baseline_df.copy()
        self.alpha = alpha
        self.threshold_flagged_ratio = threshold_flagged_ratio

        if self.baseline_df.empty:
            raise ValueError("Baseline data cannot be empty.")

    def check_drift(self, incoming_batch_df):
        # ADDED: Validate incoming batch.
        if incoming_batch_df.empty:
            raise ValueError("Incoming batch cannot be empty.")

        if list(self.baseline_df.columns) != list(incoming_batch_df.columns):
            raise ValueError(
                "Baseline and incoming batch must have identical columns "
                "in the same order."
            )

        drifted_features = []

        for col in self.baseline_df.columns:
            baseline_values = self.baseline_df[col].to_numpy()
            incoming_values = incoming_batch_df[col].to_numpy()

            # ADDED: Reject invalid values before the statistical test.
            if (
                not np.isfinite(baseline_values).all()
                or not np.isfinite(incoming_values).all()
            ):
                raise ValueError(
                    f"Feature '{col}' contains missing or infinite values."
                )

            stat, p_val = ks_2samp(
                baseline_values,
                incoming_values
            )

            if p_val < self.alpha:
                drifted_features.append(col)

        drift_ratio = (
            len(drifted_features) / len(self.baseline_df.columns)
        )

        return {
            "drift_ratio": drift_ratio,
            "drifted_features_count": len(drifted_features),
            "drifted_features": drifted_features,
            "alert": drift_ratio >= self.threshold_flagged_ratio
        }


def main():
    print("=" * 60)
    print(" AIML-02: FRAUD DETECTION & DRIFT MONITORING PIPELINE ")
    print("=" * 60)

    # Load the dataset.
    raw_df = load_or_generate_data()

    # ADDED: Check that the dataset is large enough for stratified splitting.
    if len(raw_df) < 10:
        raise ValueError(
            "Dataset is too small. Provide at least 10 transactions."
        )

    X, y, feature_names, amount_scaler, time_scaler = (
        preprocess_features(raw_df)
    )

    # CHANGED: Split before fitting the feature scalers.
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42
    )

    # ADDED: Fit scalers only on training data.
    X_train, X_val = scale_features(
        X_train,
        X_val,
        amount_scaler,
        time_scaler
    )

    print(f"\n[+] Training transactions: {len(X_train)}")
    print(f"[+] Validation transactions: {len(X_val)}")
    print(f"[+] Fraud transactions in dataset: {int(y.sum())}")

    # Train supervised classifier.
    print(
        "\n[*] Training Supervised Classifier "
        "(Class-Weighted Random Forest)..."
    )

    supervised_model = RandomForestClassifier(
        n_estimators=50,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    )

    supervised_model.fit(X_train, y_train)

    val_probs_sup = supervised_model.predict_proba(X_val)[:, 1]

    pr_auc_sup = average_precision_score(
        y_val,
        val_probs_sup
    )

    # ADDED: Calculate ROC-AUC when both classes are present.
    roc_auc_sup = roc_auc_score(y_val, val_probs_sup)

    # Train unsupervised anomaly detector.
    print(
        "\n[*] Training Unsupervised Anomaly Detector "
        "(Isolation Forest)..."
    )

    iso_forest = IsolationForest(
        n_estimators=50,
        contamination=0.002,
        random_state=42,
        n_jobs=-1
    )

    iso_forest.fit(X_train)

    # Higher anomaly scores indicate more unusual transactions.
    iso_scores = -iso_forest.decision_function(X_val)

    pr_auc_iso = average_precision_score(
        y_val,
        iso_scores
    )

    print("\n" + "=" * 50)
    print(f" Supervised PR-AUC  : {pr_auc_sup:.4f}")
    print(f" Supervised ROC-AUC : {roc_auc_sup:.4f}")
    print(f" Unsupervised PR-AUC: {pr_auc_iso:.4f}")
    print("=" * 50)

    # Financial cost optimization.
    best_t, min_c, details = evaluate_financial_cost(
        y_val,
        val_probs_sup
    )

    print("\n[*] Financial Optimization")
    print(f"    Optimal threshold : {best_t:.4f}")
    print(f"    Minimum estimated loss: ${min_c:,.2f}")
    print(
        f"    TP: {details['TP']} | "
        f"FP: {details['FP']} | "
        f"TN: {details['TN']} | "
        f"FN: {details['FN']}"
    )

    # ADDED: Report validation-set confusion matrix at the chosen threshold.
    val_predictions = (val_probs_sup >= best_t).astype(int)

    print("\n[*] Confusion Matrix at Optimized Threshold")
    print(
        confusion_matrix(
            y_val,
            val_predictions,
            labels=[0, 1]
        )
    )

    # Drift monitoring.
    print("\n[*] Running Drift Simulation...")

    monitor = DataDriftMonitor(
        baseline_df=X_train
    )

    # ADDED: Use the same batch size for both comparisons.
    batch_size = min(3000, len(X_val))

    normal_batch = X_val.sample(
        n=batch_size,
        random_state=101
    )

    status_1 = monitor.check_drift(normal_batch)

    print("\n    Batch 1 (Normal)")
    print(f"    Drifted features: {status_1['drifted_features_count']}")
    print(f"    Drift ratio: {status_1['drift_ratio']:.2%}")
    print(f"    Retrain Alert: {status_1['alert']}")

    # Simulate distribution changes in selected features.
    drifted_batch = X_val.sample(
        n=batch_size,
        random_state=202
    ).copy()

    rng = np.random.default_rng(202)

    drifted_batch["scaled_amount"] += rng.normal(
        2.0, 0.5, size=len(drifted_batch)
    )

    drifted_batch["V1"] -= rng.exponential(
        1.5, size=len(drifted_batch)
    )

    drifted_batch["V3"] += rng.normal(
        2.0, 0.4, size=len(drifted_batch)
    )

    drifted_batch["V12"] -= rng.normal(
        2.5, 0.5, size=len(drifted_batch)
    )

    drifted_batch["V14"] -= rng.normal(
        3.0, 0.6, size=len(drifted_batch)
    )

    status_2 = monitor.check_drift(drifted_batch)

    print("\n    Batch 2 (Shifted)")
    print(f"    Drifted features: {status_2['drifted_features_count']}")
    print(f"    Drift ratio: {status_2['drift_ratio']:.2%}")
    print(f"    Retrain Alert: {status_2['alert']}")

    if status_2["alert"]:
        print(
            "\n[ALERT] SIGNIFICANT DRIFT DETECTED IN: "
            + ", ".join(status_2["drifted_features"][:5])
        )
    else:
        print("\n[INFO] Drift threshold was not reached.")

    print("\n[+] Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
