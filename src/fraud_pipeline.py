import sys
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

def load_or_generate_data(file_path="data/creditcard.csv", n_samples=100000, random_state=42):
    try:
        df = pd.read_csv(file_path)
        print(f"[+] Loaded real dataset: {df.shape[0]} transactions.")
        return df
    except FileNotFoundError:
        print(f"[-] '{file_path}' not found. Generating synthetic benchmark dataset ({n_samples} rows)...")
        np.random.seed(random_state)
        n_fraud = int(n_samples * 0.0017)
        n_normal = n_samples - n_fraud

        normal_v = np.random.normal(loc=0.0, scale=1.0, size=(n_normal, 28))
        normal_amt = np.random.exponential(scale=85.0, size=(n_normal, 1))
        normal_time = np.sort(np.random.uniform(0, 172800, size=(n_normal, 1)))
        normal_labels = np.zeros((n_normal, 1))

        fraud_v = np.random.normal(loc=0.0, scale=1.0, size=(n_fraud, 28))
        fraud_v[:, 0] -= 2.5
        fraud_v[:, 2] += 3.0
        fraud_v[:, 11] -= 3.5
        fraud_amt = np.random.exponential(scale=180.0, size=(n_fraud, 1))
        fraud_time = np.sort(np.random.uniform(0, 172800, size=(n_fraud, 1)))
        fraud_labels = np.ones((n_fraud, 1))

        normal_data = np.hstack([normal_time, normal_v, normal_amt, normal_labels])
        fraud_data = np.hstack([fraud_time, fraud_v, fraud_amt, fraud_labels])
        all_data = np.vstack([normal_data, fraud_data])

        col_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
        df = pd.DataFrame(all_data, columns=col_names)
        df = df.sort_values(by="Time").reset_index(drop=True)
        df["Class"] = df["Class"].astype(int)
        return df

def preprocess_features(df):
    df_clean = df.copy()
    scaler = RobustScaler()
    df_clean["scaled_amount"] = scaler.fit_transform(df_clean["Amount"].values.reshape(-1, 1))
    df_clean["scaled_time"] = scaler.fit_transform(df_clean["Time"].values.reshape(-1, 1))
    df_clean = df_clean.drop(["Time", "Amount"], axis=1)
    
    feature_cols = [c for c in df_clean.columns if c != "Class"]
    X = df_clean[feature_cols]
    y = df_clean["Class"]
    return X, y, feature_cols

def evaluate_financial_cost(y_true, y_probs, cost_fn=150.0, cost_fp=10.0):
    thresholds = np.linspace(0.01, 0.99, 100)
    best_threshold = 0.5
    min_cost = float("inf")
    metrics_at_best = {}

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        total_cost = (fn * cost_fn) + (fp * cost_fp)

        if total_cost < min_cost:
            min_cost = total_cost
            best_threshold = t
            metrics_at_best = {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "Cost": total_cost}

    return best_threshold, min_cost, metrics_at_best

class DataDriftMonitor:
    def __init__(self, baseline_df, alpha=0.01, threshold_flagged_ratio=0.15):
        self.baseline_df = baseline_df
        self.alpha = alpha
        self.threshold_flagged_ratio = threshold_flagged_ratio

    def check_drift(self, incoming_batch_df):
        drifted_features = []
        for col in self.baseline_df.columns:
            stat, p_val = ks_2samp(self.baseline_df[col], incoming_batch_df[col])
            if p_val < self.alpha:
                drifted_features.append(col)

        drift_ratio = len(drifted_features) / len(self.baseline_df.columns)
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

    raw_df = load_or_generate_data()
    X, y, feature_names = preprocess_features(raw_df)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

    print("\n[*] Training Supervised Classifier (Class-Weighted Random Forest)...")
    supervised_model = RandomForestClassifier(n_estimators=50, class_weight="balanced_subsample", n_jobs=-1, random_state=42)
    supervised_model.fit(X_train, y_train)
    val_probs_sup = supervised_model.predict_proba(X_val)[:, 1]
    pr_auc_sup = average_precision_score(y_val, val_probs_sup)

    print("[*] Training Unsupervised Anomaly Detector (Isolation Forest)...")
    iso_forest = IsolationForest(n_estimators=50, contamination=0.002, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train)
    iso_scores = -iso_forest.decision_function(X_val)
    pr_auc_iso = average_precision_score(y_val, iso_scores)

    print("\n" + "=" * 50)
    print(f" Supervised PR-AUC : {pr_auc_sup:.4f}")
    print(f" Unsupervised PR-AUC : {pr_auc_iso:.4f}")
    print("=" * 50)

    best_t, min_c, details = evaluate_financial_cost(y_val, val_probs_sup)
    print(f"\n[*] Financial Optimization: Minimum Loss = ${details['Cost']:,.2f} at threshold {best_t:.4f}")

    print("\n[*] Running Drift Simulation...")
    monitor = DataDriftMonitor(baseline_df=X_train)
    
    normal_batch = X_val.sample(n=3000, random_state=101)
    status_1 = monitor.check_drift(normal_batch)
    print(f"    - Batch 1 (Normal) Retrain Alert: {status_1['alert']}")

    drifted_batch = X_val.sample(n=3000, random_state=202).copy()
    drifted_batch["scaled_amount"] += np.random.normal(2.0, 0.5, size=len(drifted_batch))
    drifted_batch["V1"] -= np.random.exponential(1.5, size=len(drifted_batch))
    drifted_batch["V3"] += np.random.normal(2.0, 0.4, size=len(drifted_batch))
    drifted_batch["V12"] -= np.random.normal(2.5, 0.5, size=len(drifted_batch))
    drifted_batch["V14"] -= np.random.normal(3.0, 0.6, size=len(drifted_batch))
    
    status_2 = monitor.check_drift(drifted_batch)
    print(f"    - Batch 2 (Shifted) Retrain Alert: {status_2['alert']}")

    if status_2["alert"]:
        print(f"\n[ALERT] SIGNIFICANT DRIFT DETECTED IN: {', '.join(status_2['drifted_features'][:3])}...")

if __name__ == "__main__":
    main()