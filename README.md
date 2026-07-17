# Fraud-Detection-Assignment-

Fraud Detection: From Classical Data Analysis to Neural Networks
A comprehensive fraud detection pipeline applied to synthetic financial transaction data, progressing from classical statistical methods to deep learning-based anomaly detection.

Overview
This project analyzes transaction-level financial data to identify suspicious activity, combining exploratory data analysis, regression modeling, and neural network techniques. It's structured to demonstrate a full analytical workflow — from raw data to actionable fraud-detection insights — using both supervised and unsupervised learning approaches.

The dataset consists of synthetic financial transactions (synthetic_financial_transactions.csv) designed to mimic realistic patterns without corresponding to real customers or institutions. Each row represents a single transaction with features covering:

Customer attributes: age, income band, tenure, historical transaction behavior

Transaction details: amount, merchant type, channel (POS/online/mobile), region, timing (hour, day of week, week)

Risk indicators: foreign transaction flag, card-present flag, high-risk merchant flag, night/weekend flags, internal risk score

Target variables: fraud (binary label) and loss_amount (estimated monetary loss)

Methodology
The analysis follows a structured, multi-stage workflow:

1. Data Preprocessing
Loading, cleaning, and preparing the dataset for modeling, including handling missing values, encoding categorical variables, scaling numerical features, and creating train/test splits.

2. Descriptive Statistics & Visualization
Summary statistics and visual exploration of transaction patterns, including distribution plots, fraud-rate breakdowns, and correlation analysis between numerical features.

3. Linear Regression
Modeling transaction amount as a continuous target to identify key drivers of transaction size, evaluated using RMSE and R².

4. Logistic Regression
Baseline classification model for predicting fraud, evaluated with confusion matrix, precision, recall, F1-score, and ROC-AUC — with attention to class imbalance.

5. Object-Oriented Pipeline
A reusable Python class (FraudAnalysisPipeline) encapsulating preprocessing, descriptive analysis, and model fitting/evaluation steps for clean, modular code.

6. Neural Network Classifier
A feedforward neural network (input layer, one or two hidden layers, binary output) trained to predict fraud, benchmarked against the logistic regression baseline.

7. Autoencoder Anomaly Detection
An unsupervised autoencoder trained exclusively on non-fraud transactions, using reconstruction error to flag anomalies — compared against ground-truth fraud labels to assess detection of cases missed by supervised models.

8. Final Comparison
A consolidated comparison of all four modeling approaches (linear regression, logistic regression, neural network, autoencoder), with a discussion of trade-offs and a final recommendation.

Tools & Libraries
Data handling: pandas, numpy

Visualization: matplotlib

Modeling: scikit-learn (linear/logistic regression, evaluation metrics)

Deep learning: TensorFlow/Keras or PyTorch (neural network and autoencoder)



Key Questions Addressed
Which variables best explain transaction amount and fraud risk?

Why is accuracy alone insufficient for imbalanced fraud classification?

Does the neural network meaningfully outperform logistic regression, and does that justify its added complexity?

Can an autoencoder surface suspicious transactions that supervised models miss?

Which method would be most suitable for deployment in a financial institution's fraud monitoring system?

Disclaimer
This dataset is synthetic and created for educational purposes only. It does not represent real customers, transactions, or institutions.
