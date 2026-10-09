# AIML-02: Fraud & Anomaly Detection with Drift Monitoring

A machine learning pipeline for detecting fraudulent credit card transactions and monitoring changes in transaction patterns over time.

## 1. Project Overview

Fraud detection models can become less effective as transaction patterns change, new fraud tactics emerge, and customer spending behavior evolves. A model trained once may gradually become less reliable if these changes are not monitored.

This project implements a fraud detection pipeline that combines supervised classification, unsupervised anomaly detection, financial cost optimization, and statistical data drift monitoring.

The pipeline also simulates artificially shifted transaction data to demonstrate how a drift monitoring system can identify changes in feature distributions and trigger a retraining alert.

## 2. Objectives

* Handle the extreme class imbalance found in credit card fraud datasets.
* Train and compare supervised and unsupervised machine learning models.
* Evaluate model performance using Average Precision (PR-AUC).
* Detect changes in transaction feature distributions.
* Trigger an alert when the configured drift threshold is reached.
* Analyze the financial costs of missed fraud and incorrectly flagged legitimate transactions.

## 3. Technology Stack

* **Python** — Core programming language
* **Pandas and NumPy** — Data processing and numerical operations
* **Scikit-learn** — Model training, preprocessing, and evaluation
* **SciPy** — Statistical drift detection
* **Matplotlib** — Available for visualization and plotting

## 4. Models and Methodology

### 4.1 Supervised Fraud Classification

The project uses a Random Forest classifier with class weighting.

Class weighting assigns greater importance to the minority fraud class during model training. This helps address class imbalance without relying exclusively on oversampling techniques such as SMOTE.

The model generates fraud probabilities that are evaluated using Average Precision.

### 4.2 Unsupervised Anomaly Detection

The project uses Isolation Forest to identify transactions that differ from typical transaction patterns.

Unlike the supervised classifier, Isolation Forest does not use the fraud labels during model fitting. Its anomaly scores are evaluated against the available labels to compare its ability to identify fraudulent transactions.

### 4.3 Model Evaluation

The pipeline reports Average Precision for both models.

Average Precision summarizes the precision-recall relationship across different classification thresholds. It is particularly useful for imbalanced fraud detection problems because accuracy can appear high even when a model fails to identify most fraudulent transactions.

ROC-AUC may provide additional information, but it should not be the only evaluation metric for a highly imbalanced dataset.

### 4.4 Financial Cost Optimization

The pipeline evaluates different classification thresholds using an illustrative financial cost model.

The current implementation assumes:

* False negative cost: $150 per missed fraudulent transaction.
* False positive cost: $10 per legitimate transaction incorrectly flagged as fraudulent.

The total estimated cost is calculated as:

`Total Cost = (False Negatives × Cost per False Negative) + (False Positives × Cost per False Positive)`

The pipeline selects the threshold that minimizes this estimated cost over the evaluated threshold range.

These values are assumptions for demonstration purposes, not measured costs from a real financial institution.

### 4.5 Data Drift Detection

The project uses the two-sample Kolmogorov-Smirnov (KS) test to compare feature distributions between a baseline dataset and an incoming batch.

For each feature, the test checks whether the two samples show statistically significant evidence of a distribution difference.

The configured significance level is `0.01`.

The monitor calculates the proportion of features flagged as drifted. A retraining alert is raised when the proportion reaches or exceeds `0.15` (15%).

This is a basic statistical data drift check. A drift alert indicates a change in feature distributions; it does not by itself prove that model accuracy has decreased.

### 4.6 Simulated Drift

The pipeline compares two batches:

1. **Normal batch:** Sampled from the validation data without deliberately changing its feature distributions.
2. **Shifted batch:** A copy of sampled validation data in which selected features are artificially modified.

The monitor evaluates both batches and reports whether its configured alert threshold has been reached.

This simulation demonstrates the drift detection mechanism without requiring a continuously running transaction stream.

## 5. Project Structure

```text
fraud---drift---detection/
├── src/
│   └── fraud_pipeline.py
├── data/
│   └── creditcard.csv        # Optional; not included in the repository
├── .gitignore
├── README.md
└── requirements.txt
```

The `data/` directory and dataset are optional. If the expected dataset is unavailable, the pipeline generates a synthetic dataset for demonstration.

## 6. Dataset

The pipeline supports the commonly used Credit Card Fraud Detection dataset, which contains transaction features, an `Amount` column, a `Time` column, and a binary `Class` label.

* `Class = 0`: Legitimate transaction
* `Class = 1`: Fraudulent transaction

The pipeline expects the dataset at:

`data/creditcard.csv`

The dataset is not included in this repository.

If the file is not found, the pipeline generates a synthetic dataset with a small fraud proportion. This allows the pipeline to be executed without downloading the real dataset.

**Important:** Synthetic data is intended for testing and demonstration. Performance measured on synthetic data should not be presented as verified performance on real financial transactions.

## 7. Installation and Setup

### Prerequisites

* Python 3.9 or later
* Git
* pip

### Step 1: Clone the repository

```bash
git clone https://github.com/rudrathakkar999/fraud---drift---detection.git
cd fraud---drift---detection
```

### Step 2: Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks virtual environment activation, use Command Prompt instead:

```cmd
venv\Scripts\activate.bat
```

**Linux or macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
python -m pip install -r requirements.txt
```

## 8. Running the Pipeline

From the repository root directory, execute:

```bash
python src/fraud_pipeline.py
```

The pipeline trains both models, calculates their Average Precision scores, performs financial cost optimization, and runs the drift simulation.

The output includes:

* Supervised model Average Precision
* Unsupervised model Average Precision
* Estimated minimum financial cost and selected threshold
* Drift alert status for the normal batch
* Drift alert status for the artificially shifted batch

Actual metrics and alert results depend on the dataset and execution environment.

## 9. Reproducing the Evaluation

To reproduce the pipeline's results:

1. Clone the repository.
2. Install the dependencies.
3. Use the same dataset and file path, or allow the pipeline to generate synthetic data.
4. Run `python src/fraud_pipeline.py`.
5. Record the Average Precision scores, financial cost, selected threshold, and drift alert results.

The pipeline uses fixed random seeds in key model and data-generation operations to improve reproducibility.

For comparisons between different model versions, use the same dataset and evaluation setup.

## 10. Limitations

* Synthetic data does not reproduce the full complexity of real-world fraud.
* The financial cost values are illustrative assumptions.
* The drift monitor detects feature distribution changes rather than directly measuring changes in fraud prediction accuracy.
* The drift simulation modifies selected features to demonstrate the monitoring mechanism.
* The current implementation simulates batches rather than monitoring a continuous live transaction stream.
* A drift alert indicates that further investigation or model evaluation may be needed; it does not automatically establish that retraining will improve performance.

## 11. Future Improvements

* Evaluate the models on an independent test set.
* Add precision, recall, F1-score, and confusion matrix reporting.
* Validate drift alert behavior across repeated normal and shifted batches.
* Add visualizations for model performance and drift statistics.
* Implement scheduled monitoring and a controlled model retraining workflow.

## 12. Conclusion

This project demonstrates a machine learning approach to fraud detection that combines class-weighted supervised learning, unsupervised anomaly detection, precision-recall evaluation, financial cost-based threshold selection, and statistical drift monitoring.

By combining fraud detection with a basic drift alert mechanism, the pipeline illustrates why monitoring changing transaction patterns is an important part of maintaining machine learning systems over time.
