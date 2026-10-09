# AIML-02: Fraud & Anomaly Detection with Drift Monitoring

A machine learning pipeline for credit card fraud detection, unsupervised anomaly detection, financial cost optimization, and statistical data drift monitoring.

## 1. Project Overview

Fraud detection is challenging because fraudulent transactions represent a very small proportion of all transactions. Machine learning models must identify suspicious activity while minimizing false alarms.

This project combines supervised and unsupervised machine learning with financial cost optimization and statistical drift monitoring.

## 2. Features

* Fraud classification using a class-weighted Random Forest.
* Unsupervised anomaly detection using Isolation Forest.
* Model evaluation using PR-AUC and ROC-AUC.
* Financial cost-based classification threshold optimization.
* Confusion matrix reporting.
* Feature distribution monitoring using the two-sample Kolmogorov-Smirnov (KS) test.
* Simulated data drift and automated threshold-based alerts.
* Synthetic dataset generation when the real dataset is unavailable.

## 3. Technology Stack

* Python
* NumPy
* Pandas
* Scikit-learn
* SciPy

## 4. Machine Learning Methodology

### Supervised Learning

A Random Forest classifier with `balanced_subsample` class weighting is used to classify transactions as legitimate or fraudulent.

The model generates fraud probabilities and is evaluated using Average Precision (PR-AUC) and ROC-AUC.

### Unsupervised Learning

Isolation Forest identifies unusual transaction patterns without using fraud labels during training.

Its anomaly scores are evaluated against the available transaction labels using PR-AUC.

### Financial Cost Optimization

The pipeline selects a classification threshold that minimizes the estimated financial cost over the evaluated threshold range.

The current cost assumptions are:

* False negative: $150 per missed fraud transaction.
* False positive: $10 per legitimate transaction incorrectly flagged.

The estimated total cost is:

`Total Cost = (False Negatives × $150) + (False Positives × $10)`

These costs are illustrative assumptions, not actual costs from a financial institution.

### Data Drift Monitoring

The pipeline compares baseline transaction features against incoming batches using the two-sample KS test.

Configuration:

* Significance level: 0.01
* Drift alert threshold: 15% of monitored features

An alert is raised when the proportion of drifted features reaches or exceeds the configured threshold.

The pipeline simulates both a normal batch and an artificially shifted batch to demonstrate the monitoring mechanism.

A drift alert indicates a change in feature distributions; it does not automatically mean that model performance has deteriorated.

## 5. Dataset

This project supports the Kaggle Credit Card Fraud Detection dataset.

The original dataset contains 284,807 transactions and 492 fraudulent transactions.

Dataset source:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download `creditcard.csv` and place it in the project's `data` directory.

Expected structure:

```text
AWS SBG (Student Builder Group)/
├── src/
│   └── fraud_pipeline.py
├── data/
│   └── creditcard.csv
├── README.md
├── requirements.txt
└── .gitignore
```

The dataset is not included in this repository.

If the expected dataset is unavailable, the pipeline generates a synthetic dataset for demonstration. Synthetic data is useful for testing but does not represent verified real-world fraud detection performance.

## 6. Installation

### Prerequisites

* Python 3.9 or later
* Git
* pip

### Clone the repository

```powershell
git clone https://github.com/rudrathakkar999/fraud---drift---detection.git
cd fraud---drift---detection
```

### Create a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## 7. Run the Pipeline

From the repository root, enter the source directory:

```powershell
cd src
```

Run the pipeline:

```powershell
python fraud_pipeline.py
```

The pipeline trains both models, evaluates their performance, optimizes the financial decision threshold, and runs the drift simulation.

The dataset path must match your local configuration. If the script uses `../data/creditcard.csv`, run it from the `src` directory as shown above.

## 8. Example Results

The following results were obtained during one run on the real dataset.

| Metric                        |  Result |
| ----------------------------- | ------: |
| Dataset transactions          | 284,807 |
| Training transactions         | 199,364 |
| Validation transactions       |  85,443 |
| Supervised PR-AUC             |  0.8181 |
| Supervised ROC-AUC            |  0.9248 |
| Isolation Forest PR-AUC       |  0.1759 |
| Selected financial threshold  |  0.1090 |
| Validation-set optimized cost |  $3,970 |

### Confusion Matrix

```text
[[85258    37]
 [   24   124]]
```

The model detected 124 of the 148 fraud transactions in the validation set.

These metrics are validation results, not independent estimates of performance on unseen test data. Results may vary with dataset versions and code changes.

### Drift Simulation

| Batch         | Drift ratio | Alert |
| ------------- | ----------: | ----- |
| Normal batch  |       0.00% | No    |
| Shifted batch |      16.67% | Yes   |

The shifted batch exceeded the configured 15% threshold and triggered an alert.

## 9. Limitations

* Fraud detection performance requires further evaluation on an independent test set.
* Financial costs are illustrative assumptions.
* Synthetic data does not reproduce the full complexity of real financial transactions.
* The KS test detects distribution changes, not fraud model accuracy directly.
* Drift is simulated rather than monitored from a live transaction stream.
* A drift alert does not guarantee that retraining will improve performance.

## 10. Future Improvements

* Evaluate the selected threshold on an independent test set.
* Add precision, recall, and F1-score reporting.
* Evaluate drift detection across repeated batches.
* Add visualizations for model performance and drift statistics.
* Develop a controlled model retraining workflow.

## 11. Conclusion

This project demonstrates a fraud detection pipeline that combines class-weighted supervised learning, unsupervised anomaly detection, financial cost optimization, and statistical data drift monitoring.

It illustrates how fraud detection and distribution monitoring can work together to help identify suspicious transactions and changes in transaction patterns.
