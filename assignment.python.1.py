# Create a new environment named fraud-env with Python 3.11
conda create -n fraud-env python=3.11 -y

# Activate that environment
conda activate fraud-env
conda init
# Install core scientific libraries
conda install pandas numpy matplotlib seaborn scikit-learn statsmodels -y

# Install TensorFlow (for Keras)
conda install tensorflow -y

# Install Keras via pip in this same environment
pip install keras



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score,
    confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
)

import statsmodels.api as sm

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


# -------------------------------------------------------------------
# A(a) Read the CSV file into Python
# -------------------------------------------------------------------
df = pd.read_csv("/Users/nikoslamprou/Desktop/synthetic_transactions.csv")  

# -------------------------------------------------------------------
# A(b) Display first rows and dataset dimensions
# -------------------------------------------------------------------
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset dimensions (rows, columns):")
print(df.shape)

# -------------------------------------------------------------------
# A(c) Check variable types
# -------------------------------------------------------------------
print("\nVariable types:")
print(df.dtypes)

# -------------------------------------------------------------------
# A(d) Check for missing values and decide how to handle them
# -------------------------------------------------------------------
print("\nMissing values per column:")
print(df.isna().sum())

# Example strategy:
# - If only a few missing values: impute numerics with median, categoricals with mode.
# - If there are entire columns with too many missing values, consider dropping.
# Here we do a simple, transparent approach.

# Identify numerical and categorical columns from dtypes first
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# We'll refine categoricals in A(e), but start with non-numeric:
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

print("\nInitial numeric columns:")
print(numeric_cols)
print("\nInitial categorical columns (non-numeric types):")
print(categorical_cols)

# Simple imputation:
# For numeric columns: fill missing with median
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# For categorical columns: fill missing with most frequent value
for col in categorical_cols:
    if df[col].isna().sum() > 0:
        mode_val = df[col].mode().iloc[0]
        df[col] = df[col].fillna(mode_val)

print("\nMissing values after simple imputation:")
print(df.isna().sum())






# -----------------------------
# A(e) Identify categorical vs numerical (by meaning)
# -----------------------------

# IDs and target
id_cols = ["transaction_id", "customer_id"]
target_col = "fraud"

# Categorical variables from your dtypes and assignment description
categorical_features = ["merchant_type", "channel", "region"]

# Binary indicator features (0/1 but conceptually categorical)
binary_features = [
    "is_foreign",
    "card_present",
    "high_risk_merchant",
    "night_txn",
    "weekend"
]

# Numeric / ordered variables (excluding IDs and fraud)
numeric_features = [
    "customer_age",
    "income_band",
    "tenure_months",
    "hour",
    "day_of_week",
    "week",
    "avg_prev_amount",
    "daily_txn_count",
    "days_since_last_txn",
    "amount",
    "amount_ratio",
    "risk_score",
    "loss_amount"
]

print("\nCategorical features:", categorical_features)
print("Binary features:", binary_features)
print("Numeric features:", numeric_features)

# -----------------------------
# A(f) Handle missing values with a simple strategy
# (now that we know which columns are what)
# -----------------------------

# Impute numeric columns: median
df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())

# Impute categorical columns: mode (most frequent category)
for col in categorical_features:
    if df[col].isna().sum() > 0:
        mode_val = df[col].mode().iloc[0]
        df[col] = df[col].fillna(mode_val)

print("\nMissing values after imputation:")
print(df.isna().sum())






from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np

# -----------------------------
# A(f.1) Define X and y; drop IDs and target
# -----------------------------
df_model = df.copy()

X = df_model.drop(columns=id_cols + [target_col])
y = df_model[target_col]

print("\nFeatures shape before encoding:", X.shape)

# -----------------------------
# A(f.2) Train/test split (stratified by fraud)
# -----------------------------
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

print("Train shape (raw):", X_train_raw.shape)
print("Test shape (raw):", X_test_raw.shape)

# -----------------------------
# A(f.3) One-hot encode categorical features
# -----------------------------
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

X_train_cat = ohe.fit_transform(X_train_raw[categorical_features])
X_test_cat = ohe.transform(X_test_raw[categorical_features])

ohe_feature_names = ohe.get_feature_names_out(categorical_features)

X_train_cat_df = pd.DataFrame(
    X_train_cat, columns=ohe_feature_names, index=X_train_raw.index
)
X_test_cat_df = pd.DataFrame(
    X_test_cat, columns=ohe_feature_names, index=X_test_raw.index
)

# Drop original categorical cols and add encoded versions
X_train_num = X_train_raw.drop(columns=categorical_features)
X_test_num = X_test_raw.drop(columns=categorical_features)

X_train_encoded = pd.concat([X_train_num, X_train_cat_df], axis=1)
X_test_encoded = pd.concat([X_test_num, X_test_cat_df], axis=1)

print("\nFeatures shape after encoding:")
print("X_train_encoded:", X_train_encoded.shape)
print("X_test_encoded:", X_test_encoded.shape)

# -----------------------------
# A(f.4) Scale numerical variables
# -----------------------------
scaler = StandardScaler()

# Choose which columns to scale: numeric + binary
numeric_cols_for_scaling = [
    col for col in X_train_encoded.columns
    if col in numeric_features + binary_features
]

print("\nNumeric columns to scale:", numeric_cols_for_scaling)

X_train_scaled = X_train_encoded.copy()
X_test_scaled = X_test_encoded.copy()

X_train_scaled[numeric_cols_for_scaling] = scaler.fit_transform(
    X_train_encoded[numeric_cols_for_scaling]
)
X_test_scaled[numeric_cols_for_scaling] = scaler.transform(
    X_test_encoded[numeric_cols_for_scaling]
)

print("\nSample of scaled training data:")
print(X_train_scaled.head())










# ================================
# Part B: Descriptive statistics and visualization
# ================================

import matplotlib.pyplot as plt
import seaborn as sns



# -----------------------------
# B(a) Summary statistics for numerical variables
# -----------------------------
# We compute basic descriptive statistics (count, mean, std, quantiles)
# for all numeric and binary variables, to understand their distribution.
num_cols_for_desc = numeric_features + binary_features

print("\nSummary statistics for numerical variables:")
print(df[num_cols_for_desc].describe())

# -----------------------------
# B(b) Proportion of fraudulent transactions
# -----------------------------
# Fraud is a binary variable (0/1). The mean gives the proportion of frauds.
fraud_rate = df['fraud'].mean()
print(f"\nProportion of fraudulent transactions: {fraud_rate:.4f}")

# -----------------------------
# B(c) Plots
# We create at least four informative plots:
# 1) Histogram of amount
# 2) Bar plot of fraud counts
# 3) Boxplot of amount by fraud
# 4) Count plot of merchant_type
# 5) (Optional) Correlation heatmap for numerics
# -----------------------------

# 1. Histogram of transaction amount
plt.figure(figsize=(6, 4))
sns.histplot(df['amount'], bins=40, kde=False)
plt.title("Histogram of transaction amount")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 2. Bar plot of fraud counts (class balance)
plt.figure(figsize=(4, 4))
sns.countplot(x='fraud', data=df)
plt.title("Counts of fraud vs non-fraud")
plt.xlabel("Fraud (0 = no, 1 = yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 3. Boxplot of amount by fraud status
# This shows how the distribution of amounts differs for fraud vs non-fraud.
plt.figure(figsize=(6, 4))
sns.boxplot(x='fraud', y='amount', data=df)
plt.title("Transaction amount by fraud status")
plt.xlabel("Fraud (0 = no, 1 = yes)")
plt.ylabel("Amount")
plt.tight_layout()
plt.show()

# 4. Count plot of merchant_type
# This shows which merchant categories are most common.
plt.figure(figsize=(8, 4))
sns.countplot(y='merchant_type',
              data=df,
              order=df['merchant_type'].value_counts().index)
plt.title("Transaction counts by merchant_type")
plt.xlabel("Count")
plt.ylabel("Merchant type")
plt.tight_layout()
plt.show()

# 5. Correlation heatmap for numerical variables (optional but useful)
# We examine correlations among numeric variables and with fraud.
plt.figure(figsize=(10, 8))
corr_mat = df[num_cols_for_desc + ['fraud']].corr()
sns.heatmap(corr_mat, annot=False, cmap='coolwarm', center=0)
plt.title("Correlation heatmap (numerical variables + fraud)")
plt.tight_layout()
plt.show()

# -----------------------------
# B(d) Compare suspicious vs non-suspicious transactions
# -----------------------------
# We create a small table of mean values for key variables by fraud status.
# This helps compare typical fraud vs non-fraud transactions.
group_summary = df.groupby('fraud')[
    ['amount', 'amount_ratio', 'risk_score',
     'avg_prev_amount', 'daily_txn_count']
].mean()

print("\nMean values by fraud status (0 = non-fraud, 1 = fraud):")
print(group_summary)








# ==========================================
# Correct Part C: Predict amount WITHOUT using amount itself
# ==========================================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Target for this part: amount
y_amount_train = df.loc[X_train_scaled.index, 'amount']
y_amount_test = df.loc[X_test_scaled.index, 'amount']

# Build predictor list: drop amount, loss_amount, amount_ratio
bad_for_amount = ['amount', 'loss_amount', 'amount_ratio']

predictor_cols_for_amount = [
    col for col in X_train_scaled.columns
    if col not in bad_for_amount
]

print("\nNumber of predictors for amount (no amount, no loss_amount, no amount_ratio):",
      len(predictor_cols_for_amount))

X_train_lin = X_train_scaled[predictor_cols_for_amount]
X_test_lin = X_test_scaled[predictor_cols_for_amount]

# Fit model
lin_reg_clean = LinearRegression()
lin_reg_clean.fit(X_train_lin, y_amount_train)

# Predict and evaluate
y_amount_pred_clean = lin_reg_clean.predict(X_test_lin)

rmse_amount_clean = np.sqrt(mean_squared_error(y_amount_test, y_amount_pred_clean))
r2_amount_clean = r2_score(y_amount_test, y_amount_pred_clean)


# Coefficients for interpretation
feature_names_clean = X_train_lin.columns
coef_clean = lin_regprint("\nLinear regression performance on amount (no amount, no loss_amount, no amount_ratio):")
print(f"RMSE: {rmse_amount_clean:.2f}")
print(f"R^2:  {r2_amount_clean:.3f}")



# Coefficients for interpretation
feature_names_clean = X_train_lin.columns
coef_clean = lin_reg_clean.coef_

coef_df_clean = pd.DataFrame({
    'feature': feature_names_clean,
    'coef': coef_clean
})
coef_df_clean['abs_coef'] = coef_df_clean['coef'].abs()

coef_df_clean_sorted = coef_df_clean.sort_values(by='abs_coef', ascending=False)

print("\nTop 10 features by absolute coefficient (clean model):")
print(coef_df_clean_sorted.head(10))







# ==========================================
# Part D (clean): Logistic regression without direct leakage
# ==========================================

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    f1_score, roc_auc_score
)

# 1) Build a feature set for fraud classification
#    Drop amount and loss_amount to avoid trivial separation.
bad_for_fraud = ['amount', 'loss_amount']

fraud_predictor_cols = [
    col for col in X_train_scaled.columns
    if col not in bad_for_fraud
]

print("\nNumber of predictors for fraud (no amount, no loss_amount):",
      len(fraud_predictor_cols))

X_train_clf = X_train_scaled[fraud_predictor_cols]
X_test_clf = X_test_scaled[fraud_predictor_cols]

# 2) Fit logistic regression with class_weight="balanced"
log_reg_clean = LogisticRegression(
    solver="liblinear",
    class_weight="balanced",
    max_iter=1000
)

log_reg_clean.fit(X_train_clf, y_train)

# 3) Predictions and probabilities
y_pred_clf = log_reg_clean.predict(X_test_clf)
y_proba_clf = log_reg_clean.predict_proba(X_test_clf)[:, 1]

# 4) Confusion matrix
cm_clf = confusion_matrix(y_test, y_pred_clf)
tn, fp, fn, tp = cm_clf.ravel()

print("\nConfusion matrix (rows = true, cols = predicted) [clean model]:")
print(cm_clf)
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# 5) Metrics
precision_clf = precision_score(y_test, y_pred_clf)
recall_clf = recall_score(y_test, y_pred_clf)
f1_clf = f1_score(y_test, y_pred_clf)
accuracy_clf = (tn + tp) / (tn + fp + fn + tp)
roc_auc_clf = roc_auc_score(y_test, y_proba_clf)

print("\nLogistic regression performance (fraud = 1, clean model):")
print(f"Accuracy:  {accuracy_clf:.3f}")
print(f"Precision: {precision_clf:.3f}")
print(f"Recall:    {recall_clf:.3f}")
print(f"F1-score:  {f1_clf:.3f}")
print(f"ROC-AUC:   {roc_auc_clf:.3f}")

# 6) Coefficients for interpretation
log_coef_clean = log_reg_clean.coef_[0]
log_feature_names_clean = X_train_clf.columns

log_coef_df_clean = pd.DataFrame({
    "feature": log_feature_names_clean,
    "coef": log_coef_clean
})
log_coef_df_clean["abs_coef"] = log_coef_df_clean["coef"].abs()

log_coef_df_clean_sorted = log_coef_df_clean.sort_values(
    by="abs_coef", ascending=False
)

print("\nTop 15 features by absolute logistic coefficient (clean model, fraud risk):")
print(log_coef_df_clean_sorted.head(15))


















# ================================
# Part F: Neural network classifier for fraud (Keras)
# ================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 1) Use the same cleaned predictors as in logistic regression
bad_for_fraud = ['amount', 'loss_amount']
fraud_predictor_cols = [
    col for col in X_train_scaled.columns
    if col not in bad_for_fraud
]

X_train_nn = X_train_scaled[fraud_predictor_cols].values
X_test_nn = X_test_scaled[fraud_predictor_cols].values

input_dim = X_train_nn.shape[1]
print("NN input dimension:", input_dim)

# 2) Build a simple feedforward neural network
#    Architecture: Input -> Dense(64, relu) -> Dense(32, relu) -> Dense(1, sigmoid)
nn_model = Sequential()
nn_model.add(Dense(64, activation='relu', input_dim=input_dim))
nn_model.add(Dense(32, activation='relu'))
nn_model.add(Dense(1, activation='sigmoid'))  # binary output: probability of fraud

# 3) Compile the model
#    - Loss: binary cross-entropy for binary classification
#    - Optimizer: Adam
#    - Metrics: accuracy (we will compute precision/recall outside)
nn_model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

nn_model.summary()

# 4) Train the model
#    Note: class imbalance is strong; we can pass class weights similar to "balanced"
fraud_ratio = y_train.mean()
w_pos = 0.5 / fraud_ratio         # weight for class 1
w_neg = 0.5 / (1 - fraud_ratio)   # weight for class 0
class_weights = {0: w_neg, 1: w_pos}
print("Class weights:", class_weights)

history = nn_model.fit(
    X_train_nn,
    y_train.values,
    epochs=20,
    batch_size=256,
    validation_split=0.2,
    class_weight=class_weights,
    verbose=1
)

# 5) Evaluate on test set
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    f1_score, roc_auc_score
)
import numpy as np

# Predicted probabilities and classes
y_proba_nn = nn_model.predict(X_test_nn).ravel()
y_pred_nn = (y_proba_nn >= 0.5).astype(int)

cm_nn = confusion_matrix(y_test, y_pred_nn)
tn, fp, fn, tp = cm_nn.ravel()

print("\n[Neural network] Confusion matrix (rows = true, cols = predicted):")
print(cm_nn)
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

precision_nn = precision_score(y_test, y_pred_nn)
recall_nn = recall_score(y_test, y_pred_nn)
f1_nn = f1_score(y_test, y_pred_nn)
accuracy_nn = (tn + tp) / (tn + fp + fn + tp)
roc_auc_nn = roc_auc_score(y_test, y_proba_nn)

print("\n[Neural network] Performance (fraud = 1):")
print(f"Accuracy:  {accuracy_nn:.3f}")
print(f"Precision: {precision_nn:.3f}")
print(f"Recall:    {recall_nn:.3f}")
print(f"F1-score:  {f1_nn:.3f}")
print(f"ROC-AUC:   {roc_auc_nn:.3f}")






# ================================
# Part G: Autoencoder for anomaly detection
# ================================

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np

# 1) Use the same fraud_predictor_cols as in the classifier
X_train_ae_full = X_train_scaled[fraud_predictor_cols]
X_test_ae_full = X_test_scaled[fraud_predictor_cols]

# Split into normal (non-fraud) and fraud for training the autoencoder
train_normal_idx = y_train[y_train == 0].index
X_train_ae = X_train_ae_full.loc[train_normal_idx].values

print("Autoencoder training data shape (non-fraud only):", X_train_ae.shape)

input_dim_ae = X_train_ae.shape[1]

# 2) Define autoencoder architecture
#    Simple symmetric encoder-decoder with bottleneck
input_layer = Input(shape=(input_dim_ae,))
encoded = Dense(32, activation='relu')(input_layer)
encoded = Dense(16, activation='relu')(encoded)   # bottleneck

decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(input_dim_ae, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)

autoencoder.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse'
)

autoencoder.summary()

# 3) Train autoencoder to reconstruct normal transactions
history_ae = autoencoder.fit(
    X_train_ae, X_train_ae,
    epochs=30,
    batch_size=256,
    shuffle=True,
    validation_split=0.1,
    verbose=1
)

# 4) Compute reconstruction errors on test set (all transactions)
X_test_ae = X_test_ae_full.values
X_test_recon = autoencoder.predict(X_test_ae)

# Mean squared reconstruction error per observation
recon_errors = np.mean(np.square(X_test_ae - X_test_recon), axis=1)

# 5) Choose threshold for anomalies
#    Example: 99th percentile of reconstruction error on *normal* training data
X_train_ae_recon = autoencoder.predict(X_train_ae)
train_errors = np.mean(np.square(X_train_ae - X_train_ae_recon), axis=1)

threshold = np.percentile(train_errors, 99)
print("Chosen anomaly threshold (99th percentile of train normal errors):", threshold)

# 6) Flag anomalies in test set
anomaly_flags = (recon_errors > threshold).astype(int)

# 7) Compare anomaly flags with true fraud labels on test set
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

cm_ae = confusion_matrix(y_test, anomaly_flags)
tn_ae, fp_ae, fn_ae, tp_ae = cm_ae.ravel()

print("\n[Autoencoder] Confusion matrix (rows = true, cols = predicted anomaly):")
print(cm_ae)
print(f"TN={tn_ae}, FP={fp_ae}, FN={fn_ae}, TP={tp_ae}")

precision_ae = precision_score(y_test, anomaly_flags)
recall_ae = recall_score(y_test, anomaly_flags)
f1_ae = f1_score(y_test, anomaly_flags)

print("\n[Autoencoder] Performance (fraud vs anomaly flag):")
print(f"Precision: {precision_ae:.3f}")
print(f"Recall:    {recall_ae:.3f}")
print(f"F1-score:  {f1_ae:.3f}")




























# ================================
# Part E: FraudAnalysisPipeline class
# ================================

class FraudAnalysisPipeline:
    """
    A reusable pipeline wrapping all assignment parts:
    preprocessing, descriptive stats, linear regression,
    logistic regression, neural network, and autoencoder.
    """

    def __init__(self, df):
        # Store raw dataframe and initialise all attributes to None
        self.df = df.copy()
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.ohe = None
        self.fraud_predictor_cols = None
        self.lin_reg = None
        self.log_reg = None
        self.nn_model = None
        self.autoencoder = None
        self.lin_metrics = {}
        self.log_metrics = {}
        self.nn_metrics = {}
        self.ae_metrics = {}

    # ------------------------------------------------------------------
    def preprocess(self, test_size=0.3, random_state=42):
        """
        Impute missing values, one-hot encode categoricals,
        scale numerics, and create train/test split.
        """
        df = self.df

        id_cols = ['transaction_id', 'customer_id']
        target_col = 'fraud'
        categorical_features = ['merchant_type', 'channel', 'region']
        binary_features = [
            'is_foreign', 'card_present', 'high_risk_merchant',
            'night_txn', 'weekend'
        ]
        numeric_features = [
            'customer_age', 'income_band', 'tenure_months', 'hour',
            'day_of_week', 'week', 'avg_prev_amount', 'daily_txn_count',
            'days_since_last_txn', 'amount', 'amount_ratio',
            'risk_score', 'loss_amount'
        ]

        # Impute missing values
        df[numeric_features] = df[numeric_features].fillna(
            df[numeric_features].median()
        )
        for col in categorical_features:
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna(df[col].mode().iloc[0])

        # Build X and y
        X = df.drop(columns=id_cols + [target_col])
        y = df[target_col]

        # Stratified train/test split
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=test_size,
            stratify=y, random_state=random_state
        )

        # One-hot encode categorical features
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_train_cat = ohe.fit_transform(X_train_raw[categorical_features])
        X_test_cat = ohe.transform(X_test_raw[categorical_features])
        ohe_names = ohe.get_feature_names_out(categorical_features)

        X_train_cat_df = pd.DataFrame(
            X_train_cat, columns=ohe_names, index=X_train_raw.index
        )
        X_test_cat_df = pd.DataFrame(
            X_test_cat, columns=ohe_names, index=X_test_raw.index
        )

        X_train_enc = pd.concat(
            [X_train_raw.drop(columns=categorical_features), X_train_cat_df], axis=1
        )
        X_test_enc = pd.concat(
            [X_test_raw.drop(columns=categorical_features), X_test_cat_df], axis=1
        )

        # Scale numeric and binary features
        scaler = StandardScaler()
        cols_to_scale = [
            c for c in X_train_enc.columns
            if c in numeric_features + binary_features
        ]

        X_train_scaled = X_train_enc.copy()
        X_test_scaled = X_test_enc.copy()
        X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train_enc[cols_to_scale])
        X_test_scaled[cols_to_scale] = scaler.transform(X_test_enc[cols_to_scale])

        # Store all as class attributes
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        self.scaler = scaler
        self.ohe = ohe
        self.fraud_predictor_cols = [
            c for c in X_train_scaled.columns
            if c not in ['amount', 'loss_amount']
        ]

        print("Preprocessing complete.")
        print(f"  Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
        print(f"  Fraud rate in train: {y_train.mean():.4f}")

    # ------------------------------------------------------------------
    def descriptive_statistics(self):
        """
        Print summary statistics and produce key plots.
        """
        df = self.df
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in ['transaction_id', 'customer_id', 'fraud']
        ]

        print("\nSummary statistics:")
        print(df[numeric_cols].describe())

        print(f"\nFraud rate: {df['fraud'].mean():.4f}")

        print("\nMean values by fraud status:")
        print(df.groupby('fraud')[
            ['amount', 'amount_ratio', 'risk_score',
             'avg_prev_amount', 'daily_txn_count']
        ].mean())

        # Bar plot of fraud counts
        plt.figure(figsize=(4, 4))
        sns.countplot(x='fraud', data=df)
        plt.title("Fraud vs non-fraud counts")
        plt.tight_layout()
        plt.show()

        # Histogram of amount
        plt.figure(figsize=(6, 4))
        sns.histplot(df['amount'], bins=40)
        plt.title("Histogram of transaction amount")
        plt.tight_layout()
        plt.show()

        # Boxplot of amount by fraud status
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='fraud', y='amount', data=df)
        plt.title("Transaction amount by fraud status")
        plt.tight_layout()
        plt.show()

        # Count plot by merchant type
        plt.figure(figsize=(8, 4))
        sns.countplot(
            y='merchant_type', data=df,
            order=df['merchant_type'].value_counts().index
        )
        plt.title("Transaction counts by merchant_type")
        plt.tight_layout()
        plt.show()

        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            df[numeric_cols + ['fraud']].corr(),
            annot=False, cmap='coolwarm', center=0
        )
        plt.title("Correlation heatmap")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    def fit_linear_model(self):
        """
        Fit linear regression to predict amount (no leakage features).
        """
        bad = ['amount', 'amount_ratio', 'loss_amount']
        predictor_cols = [c for c in self.X_train_scaled.columns if c not in bad]

        y_train_amount = self.df.loc[self.X_train_scaled.index, 'amount']
        y_test_amount = self.df.loc[self.X_test_scaled.index, 'amount']

        self.lin_reg = LinearRegression()
        self.lin_reg.fit(self.X_train_scaled[predictor_cols], y_train_amount)

        y_pred = self.lin_reg.predict(self.X_test_scaled[predictor_cols])
        rmse = np.sqrt(mean_squared_error(y_test_amount, y_pred))
        r2 = r2_score(y_test_amount, y_pred)

        self.lin_metrics = {'RMSE': rmse, 'R2': r2}
        print(f"\n[Linear Regression] RMSE: {rmse:.2f}, R2: {r2:.3f}")

        coef_df = pd.DataFrame({
            'feature': predictor_cols,
            'coef': self.lin_reg.coef_
        }).sort_values(by='coef', key=abs, ascending=False)
        print("\nTop 10 coefficients:")
        print(coef_df.head(10))

    # ------------------------------------------------------------------
    def fit_logistic_model(self):
        """
        Fit logistic regression to classify fraud (no leakage features).
        """
        X_train = self.X_train_scaled[self.fraud_predictor_cols]
        X_test = self.X_test_scaled[self.fraud_predictor_cols]

        self.log_reg = LogisticRegression(
            solver='liblinear', class_weight='balanced', max_iter=1000
        )
        self.log_reg.fit(X_train, self.y_train)

        y_pred = self.log_reg.predict(X_test)
        y_proba = self.log_reg.predict_proba(X_test)[:, 1]

        self._store_clf_metrics(
            self.y_test, y_pred, y_proba,
            self.log_metrics, 'Logistic Regression'
        )

    # ------------------------------------------------------------------
    def fit_nn_classifier(self, epochs=20, batch_size=256):
        """
        Fit a feedforward neural network to classify fraud.
        """
        X_train = self.X_train_scaled[self.fraud_predictor_cols].values
        X_test = self.X_test_scaled[self.fraud_predictor_cols].values
        input_dim = X_train.shape[1]

        fraud_ratio = self.y_train.mean()
        class_weights = {0: 0.5 / (1 - fraud_ratio), 1: 0.5 / fraud_ratio}

        self.nn_model = Sequential([
            Input(shape=(input_dim,)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.nn_model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        self.nn_model.fit(
            X_train, self.y_train.values,
            epochs=epochs, batch_size=batch_size,
            validation_split=0.2,
            class_weight=class_weights,
            verbose=0
        )

        y_proba = self.nn_model.predict(X_test, verbose=0).ravel()
        y_pred = (y_proba >= 0.5).astype(int)

        self._store_clf_metrics(
            self.y_test, y_pred, y_proba,
            self.nn_metrics, 'Neural Network'
        )

    # ------------------------------------------------------------------
    def fit_autoencoder(self, epochs=30, batch_size=256, threshold_pct=99):
        """
        Train autoencoder on normal transactions;
        flag test anomalies using reconstruction error.
        """
        X_train_full = self.X_train_scaled[self.fraud_predictor_cols]
        X_test_full = self.X_test_scaled[self.fraud_predictor_cols]

        normal_idx = self.y_train[self.y_train == 0].index
        X_train_normal = X_train_full.loc[normal_idx].values
        X_test = X_test_full.values
        input_dim = X_train_normal.shape[1]

        inp = Input(shape=(input_dim,))
        enc = Dense(32, activation='relu')(inp)
        enc = Dense(16, activation='relu')(enc)
        dec = Dense(32, activation='relu')(enc)
        dec = Dense(input_dim, activation='linear')(dec)

        self.autoencoder = Model(inputs=inp, outputs=dec)
        self.autoencoder.compile(optimizer=Adam(0.001), loss='mse')
        self.autoencoder.fit(
            X_train_normal, X_train_normal,
            epochs=epochs, batch_size=batch_size,
            shuffle=True, validation_split=0.1, verbose=0
        )

        # Threshold from training reconstruction errors
        train_recon = self.autoencoder.predict(X_train_normal, verbose=0)
        train_errors = np.mean(np.square(X_train_normal - train_recon), axis=1)
        threshold = np.percentile(train_errors, threshold_pct)
        print(f"\n[Autoencoder] Threshold ({threshold_pct}th pct): {threshold:.5f}")

        # Test reconstruction errors and anomaly flags
        test_recon = self.autoencoder.predict(X_test, verbose=0)
        test_errors = np.mean(np.square(X_test - test_recon), axis=1)
        anomaly_flags = (test_errors > threshold).astype(int)

        cm = confusion_matrix(self.y_test, anomaly_flags)
        tn, fp, fn, tp = cm.ravel()
        precision = precision_score(self.y_test, anomaly_flags, zero_division=0)
        recall = recall_score(self.y_test, anomaly_flags, zero_division=0)
        f1 = f1_score(self.y_test, anomaly_flags, zero_division=0)

        self.ae_metrics = {
            'Precision': precision, 'Recall': recall, 'F1': f1,
            'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
        }

        print(f"[Autoencoder] CM:\n{cm}")
        print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

    # ------------------------------------------------------------------
    def evaluate_model(self):
        """
        Print final comparison table of all models.
        """
        print("\n========== FINAL MODEL COMPARISON ==========")
        print(f"{'Model':<25} {'Precision':>10} {'Recall':>10} "
              f"{'F1':>10} {'ROC-AUC':>10}")
        print("-" * 60)

        for label, metrics in [
            ('Logistic Regression', self.log_metrics),
            ('Neural Network', self.nn_metrics),
        ]:
            if metrics:
                print(
                    f"{label:<25}"
                    f"{metrics.get('Precision', float('nan')):>10.3f}"
                    f"{metrics.get('Recall', float('nan')):>10.3f}"
                    f"{metrics.get('F1', float('nan')):>10.3f}"
                    f"{metrics.get('ROC-AUC', float('nan')):>10.3f}"
                )

        if self.ae_metrics:
            ae = self.ae_metrics
            print(
                f"{'Autoencoder':<25}"
                f"{ae.get('Precision', float('nan')):>10.3f}"
                f"{ae.get('Recall', float('nan')):>10.3f}"
                f"{ae.get('F1', float('nan')):>10.3f}"
                f"{'N/A':>10}"
            )

        print("\n[Linear Regression on amount]")
        if self.lin_metrics:
            print(f"  RMSE: {self.lin_metrics['RMSE']:.2f}")
            print(f"  R²:   {self.lin_metrics['R2']:.3f}")

    # ------------------------------------------------------------------
    def _store_clf_metrics(self, y_true, y_pred, y_proba, store_dict, label):
        """
        Internal helper: compute, print and store classification metrics.
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = (tn + tp) / len(y_true)
        roc_auc = roc_auc_score(y_true, y_proba)

        store_dict.update({
            'Accuracy': accuracy, 'Precision': precision,
            'Recall': recall, 'F1': f1, 'ROC-AUC': roc_auc,
            'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
        })

        print(f"\n[{label}] CM:\n{cm}")
        print(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"  Accuracy: {accuracy:.3f}  Precision: {precision:.3f}  "
              f"Recall: {recall:.3f}  F1: {f1:.3f}  ROC-AUC: {roc_auc:.3f}")




# Instantiate and run the full pipeline
pipeline = FraudAnalysisPipeline(df)

pipeline.preprocess()
pipeline.descriptive_statistics()
pipeline.fit_linear_model()
pipeline.fit_logistic_model()
pipeline.fit_nn_classifier(epochs=20)
pipeline.fit_autoencoder(epochs=30)
pipeline.evaluate_model()








# ================================
# Part H: Final comparison and reflection
# ================================

# ------------------------------------------------------------------
# H(a) Summarize results from all models
# ------------------------------------------------------------------

print("=" * 65)
print("PART H: FINAL COMPARISON AND REFLECTION")
print("=" * 65)

print("""
Four modelling approaches were applied to the fraud detection problem:

1. Linear Regression: used to model the continuous variable 'amount'.
   It provides a baseline understanding of which features explain
   transaction size, but cannot be used for fraud classification
   directly since fraud is a binary outcome.

2. Logistic Regression: a supervised classifier for fraud.
   With class balancing it achieved perfect separation on this
   synthetic dataset, driven by risk_score, amount_ratio, and
   engineered indicators like night_txn and high_risk_merchant.

3. Neural Network Classifier: a feedforward NN (64-32-1 architecture)
   trained with class weights to handle imbalance.
   It achieved near-perfect results, missing only 1 fraud case
   in the test set (Recall = 0.987, F1 = 0.993).

4. Autoencoder Anomaly Detector: trained only on non-fraud data.
   It learns to reconstruct normal transactions and flags those
   with high reconstruction error as anomalies.
   Results: Precision = 0.626, Recall = 0.935, F1 = 0.750.
   This approach does not require fraud labels at training time,
   making it useful in real-world settings where labels are scarce.
""")

# ------------------------------------------------------------------
# H(b) Final comparison table
# ------------------------------------------------------------------

print("-" * 65)
print(f"{'Model':<28} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
print("-" * 65)

# Linear regression is not a classifier so we print it separately
print(f"{'Linear Regression (amount)':<28} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
print(f"  -> RMSE: {pipeline.lin_metrics['RMSE']:.2f}, R2: {pipeline.lin_metrics['R2']:.3f}")

# Supervised classifiers
for label, metrics in [
    ('Logistic Regression', pipeline.log_metrics),
    ('Neural Network',      pipeline.nn_metrics),
]:
    print(
        f"{label:<28}"
        f"{metrics.get('Precision', float('nan')):>10.3f}"
        f"{metrics.get('Recall',    float('nan')):>10.3f}"
        f"{metrics.get('F1',        float('nan')):>10.3f}"
        f"{metrics.get('ROC-AUC',   float('nan')):>10.3f}"
    )

# Autoencoder (no ROC-AUC since it is unsupervised)
ae = pipeline.ae_metrics
print(
    f"{'Autoencoder':<28}"
    f"{ae.get('Precision', float('nan')):>10.3f}"
    f"{ae.get('Recall',    float('nan')):>10.3f}"
    f"{ae.get('F1',        float('nan')):>10.3f}"
    f"{'N/A':>10}"
)
print("-" * 65)

# ------------------------------------------------------------------
# H(c) Recommendation to a financial institution
# ------------------------------------------------------------------

print("""
RECOMMENDATION:
---------------
For a real financial institution, we recommend a LAYERED approach:

1. Logistic Regression as a fast, transparent baseline.
   - Coefficients are directly interpretable (key for compliance).
   - Easy to audit and explain to regulators.
   - Low computational cost for real-time scoring.

2. Neural Network as a second-layer classifier.
   - Captures non-linear interactions between features.
   - Slightly better recall than logistic regression in more
     complex, less separable real-world datasets.
   - Trade-off: less interpretable, higher maintenance cost.

3. Autoencoder as an unsupervised complementary tool.
   - Does NOT require fraud labels to train (valuable when
     labels are noisy, delayed, or scarce in production).
   - Higher recall (0.935) means it catches most frauds,
     at the cost of more false alarms (lower precision: 0.626).
   - Best used as a first screening layer to flag suspicious
     cases for manual review, rather than as a final decision.

Important caveat:
All models show near-perfect or perfect metrics because this is a
SYNTHETIC dataset, deliberately constructed so that fraud is
separable via engineered features (risk_score, amount_ratio, etc.).
In real transactional data, performance would be lower and the
autoencoder's ability to detect unseen fraud patterns without
relying on labels would become much more valuable.
""")
