#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:25:54 2026

@author: nikoslamprou
"""

# -*- coding: utf-8 -*-
"""
Exercise 1: Causal Inference with Autoencoders (Matrix Completion)
macOS compatible — auto-installs packages, uses macOS-safe paths
"""

# =============================================================================
# AUTO-INSTALL REQUIRED PACKAGES
# =============================================================================
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

required = ["numpy", "pandas", "matplotlib", "seaborn", "scikit-learn", "tensorflow"]
for pkg in required:
    try:
        __import__(pkg if pkg != "scikit-learn" else "sklearn")
    except ImportError:
        print(f"Installing {pkg}...")
        install(pkg)

print("All packages ready.\n")

# =============================================================================
# IMPORTS
# =============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # macOS-safe non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# =============================================================================
# 1. Load and sort data
# =============================================================================
df = pd.read_csv("/Users/nikoslamprou/Desktop/MScBusinessEcon/python/final.assignment/causal_panel_data.csv")
df = df.sort_values(['unit_id', 'time']).reset_index(drop=True)

time_per_unit = df.groupby('unit_id')['time'].nunique()
assert (time_per_unit == 24).all(), "Not all units have 24 time periods"

N_units   = df['unit_id'].nunique()
T_periods = 24


# CRITICAL FIX 1: proportion ever-treated should be at unit level
prop_ever_treated = df[['unit_id', 'ever_treated']].drop_duplicates()['ever_treated'].mean()

print(f"Number of units: {N_units}")
print(f"Number of time periods: {T_periods}")
print(f"Proportion ever-treated: {prop_ever_treated:.3f}")


# =============================================================================
# 2. Wide outcome matrix Y (N x T)
# =============================================================================
Y_wide = df.pivot(index='unit_id', columns='time', values='outcome').sort_index()
Y      = Y_wide.values.astype(np.float32)
print(f"\n1.4  Wide matrix shape: {Y.shape}")

# =============================================================================
# 3. Mask matrix M (1 = observed, 0 = treated post-treatment)
# =============================================================================
treat_time_dict = (df[['unit_id', 'treat_time']]
                   .drop_duplicates()
                   .set_index('unit_id')['treat_time']
                   .to_dict())

M = np.ones_like(Y)
for i, unit in enumerate(Y_wide.index):
    t_treat = treat_time_dict.get(unit, np.nan)
    if not np.isnan(t_treat):
        col_times = Y_wide.columns.values
        M[i, col_times >= t_treat] = 0.0

n_treated_post = int(np.sum(M == 0))
print(f"\n1.2  Treated post-treatment observations: {n_treated_post}")
print(  "1.5  Masking rule: Mit=0 for treated units at and after treat_time.")
print(  "     These are the missing Y(0) values imputed by the autoencoder.")

# =============================================================================
# 4. Temporary fill: replace masked entries with 0
# =============================================================================
Y_fill         = Y.copy()
Y_fill[M == 0] = 0.0

# =============================================================================
# 5. Covariates: one-hot region/sector + standardised continuous
# =============================================================================
unit_covars = (df[['unit_id', 'region', 'sector', 'size',
                    'digital_index', 'productivity_index', 'credit_score']]
               .drop_duplicates()
               .set_index('unit_id')
               .loc[Y_wide.index])

region_dummies = pd.get_dummies(unit_covars['region'], prefix='region')
sector_dummies = pd.get_dummies(unit_covars['sector'], prefix='sector')

continuous_vars = ['size', 'digital_index', 'productivity_index', 'credit_score']
scaler_cont     = StandardScaler()
cont_scaled     = scaler_cont.fit_transform(unit_covars[continuous_vars])
cont_scaled_df  = pd.DataFrame(cont_scaled, columns=continuous_vars,
                                index=unit_covars.index)

covariates_matrix = pd.concat([region_dummies, sector_dummies,
                                cont_scaled_df], axis=1).values

# =============================================================================
# 6. Standardise outcome columns (observed values only)
# =============================================================================
Y_mean   = np.zeros(T_periods)
Y_stddev = np.ones(T_periods)
Y_std    = np.zeros_like(Y)

for t in range(T_periods):
    obs = Y[:, t][M[:, t] == 1]
    if len(obs) > 0:
        m, s         = obs.mean(), obs.std()
        s            = s if s > 1e-6 else 1.0
        Y_mean[t]    = m
        Y_stddev[t]  = s
        Y_std[:, t]  = (Y[:, t] - m) / s

# Standardise masked-outcome part of X
X_masked_std = np.zeros_like(Y_fill)
for t in range(T_periods):
    X_masked_std[:, t] = (Y_fill[:, t] - Y_mean[t]) / Y_stddev[t]

# FIX 2 — re-zero masked slots AFTER standardisation to prevent data leakage
X_masked_std[M == 0] = 0.0

# Build input matrix X = [standardised masked outcomes | mask | covariates]
X_std = np.hstack([X_masked_std, M, covariates_matrix]).astype(np.float32)
Y_std = Y_std.astype(np.float32)
M_tf  = M.astype(np.float32)

input_dim  = X_std.shape[1]
output_dim = T_periods


# -----------------------------------------------------------------------------
# 7. Build autoencoder (simple fully connected)
# -----------------------------------------------------------------------------
def build_autoencoder(input_dim, output_dim):
    input_layer = Input(shape=(input_dim,))
    dense1 = Dense(64, activation='relu')(input_layer)
    bottleneck = Dense(16, activation='relu')(dense1)
    dense2 = Dense(64, activation='relu')(bottleneck)
    output_layer = Dense(output_dim, activation='linear')(dense2)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

autoencoder = build_autoencoder(input_dim, output_dim)
optimizer = Adam(learning_rate=0.001)
# -----------------------------------------------------------------------------
# 8. Custom training loop with masked loss
# -----------------------------------------------------------------------------
def masked_mse(y_true, y_pred, mask):
    sq_diff = tf.square(y_true - y_pred)
    masked_sq_diff = sq_diff * mask
    loss = tf.reduce_sum(masked_sq_diff) / (tf.reduce_sum(mask) + 1e-8)
    return loss

batch_size = 32
epochs = 100
patience = 10
best_loss = np.inf
wait = 0
best_weights = None

print("\nTraining autoencoder with masked loss...")
for epoch in range(epochs):
    indices = np.random.permutation(X_std.shape[0])
    epoch_loss = 0.0
    n_batches = 0

    for start in range(0, X_std.shape[0], batch_size):
        end = min(start + batch_size, X_std.shape[0])
        batch_idx = indices[start:end]
        X_batch = X_std[batch_idx]
        Y_batch = Y_std[batch_idx]
        M_batch = M[batch_idx]

        with tf.GradientTape() as tape:
            Y_pred = autoencoder(X_batch, training=True)
            loss = masked_mse(Y_batch, Y_pred, M_batch)

        grads = tape.gradient(loss, autoencoder.trainable_variables)
        optimizer.apply_gradients(zip(grads, autoencoder.trainable_variables))
        epoch_loss += loss.numpy()
        n_batches += 1

    avg_loss = epoch_loss / n_batches
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        wait = 0
        best_weights = autoencoder.get_weights()
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load best weights from memory
if best_weights is not None:
    autoencoder.set_weights(best_weights)

print(f"Final training loss (masked MSE): {best_loss:.6f}")

# -----------------------------------------------------------------------------
# 9. Predict missing no-treatment outcomes
# -----------------------------------------------------------------------------
Y_pred_std = autoencoder.predict(X_std, verbose=0)
Y_pred = np.zeros_like(Y_pred_std)
for t in range(T_periods):
    Y_pred[:, t] = Y_pred_std[:, t] * Y_stddev[t] + Y_mean[t]

Yhat0 = Y_pred.copy()


# -----------------------------------------------------------------------------
# 10. Estimate treatment effects for treated post observations
# -----------------------------------------------------------------------------
units_list = Y_wide.index.tolist()
times_list = Y_wide.columns.tolist()
long_results = []

for i, unit in enumerate(units_list):
    for t_idx, time in enumerate(times_list):
        pred0 = Yhat0[i, t_idx]
        mask_val = M[i, t_idx]
        treated_post = (mask_val == 0)
        long_results.append({
            'unit_id': unit,
            'time': time,
            'pred0': pred0,
            'treated_post': treated_post
        })

df_long = pd.DataFrame(long_results)

# Drop old columns if they already exist
df = df.drop(columns=['pred0', 'treated_post'], errors='ignore')

# Merge the new columns cleanly
df = df.merge(df_long, on=['unit_id', 'time'], how='left')

# Compute treatment effects
df['tau'] = np.where(df['treated_post'], df['outcome'] - df['pred0'], np.nan)
# -----------------------------------------------------------------------------
# 11. Compute ATT estimates
# -----------------------------------------------------------------------------
overall_att = df[df['treated_post']]['tau'].mean()
print(f"\nOverall ATT: {overall_att:.6f}")

event_att = df[df['event_time'] >= 0].groupby('event_time')['tau'].mean().dropna()
print("\nEvent-time ATT:")
print(event_att)

region_att = df[df['treated_post']].groupby('region')['tau'].mean().dropna()
print("\nRegion-specific ATT:")
print(region_att)

sector_att = df[df['treated_post']].groupby('sector')['tau'].mean().dropna()
print("\nSector-specific ATT:")
print(sector_att)

med_digital = df[['unit_id', 'digital_index']].drop_duplicates()['digital_index'].median()
high_digital_att = df[(df['treated_post']) & (df['digital_index'] > med_digital)]['tau'].mean()
low_digital_att = df[(df['treated_post']) & (df['digital_index'] <= med_digital)]['tau'].mean()

print(f"\nATT for high digital (> median): {high_digital_att:.6f}")
print(f"ATT for low digital (<= median): {low_digital_att:.6f}")

# -----------------------------------------------------------------------------
# 12. Bootstrap confidence intervals
# -----------------------------------------------------------------------------
B = 50
np.random.seed(42)

boot_overall = []
boot_region = {r: [] for r in df['region'].dropna().unique()}
boot_high_digital = []
boot_low_digital = []

print(f"\nRunning bootstrap with B={B} replications...")

unit_meta = df[['unit_id', 'region', 'sector', 'size', 'digital_index',
                'productivity_index', 'credit_score', 'treat_time']].drop_duplicates().set_index('unit_id')

for b in range(B):
    boot_ids = np.random.choice(units_list, size=len(units_list), replace=True)

    boot_rows = []
    for uid in boot_ids:
        boot_rows.append(df[df['unit_id'] == uid].copy())
    df_b = pd.concat(boot_rows, ignore_index=True)
    df_b['boot_unit'] = np.repeat(np.arange(len(boot_ids)), T_periods)

    Yb_wide = df_b.pivot(index='boot_unit', columns='time', values='outcome').sort_index()
    Yb = Yb_wide.values.astype(np.float32)

    Mb = np.ones_like(Yb, dtype=float)
    for i, uid in enumerate(boot_ids):
        t_treat = treat_time_dict.get(uid, np.inf)
        if not np.isnan(t_treat):
            times = Yb_wide.columns.values
            Mb[i, times >= t_treat] = 0.0

    Yb_fill = Yb.copy()
    Yb_fill[Mb == 0] = 0.0

    b_meta = unit_meta.loc[boot_ids].reset_index(drop=True)

    region_dummies_b = pd.get_dummies(b_meta['region'], prefix='region').reindex(columns=region_dummies.columns, fill_value=0)
    sector_dummies_b = pd.get_dummies(b_meta['sector'], prefix='sector').reindex(columns=sector_dummies.columns, fill_value=0)

    cont_scaled_b = scaler_cont.transform(b_meta[continuous_vars])
    covariates_matrix_b = np.hstack([region_dummies_b.values, sector_dummies_b.values, cont_scaled_b])

    Y_std_b = np.zeros_like(Yb)
    Y_mean_b = np.zeros(T_periods)
    Y_stddev_b = np.ones(T_periods)

    for t in range(T_periods):
        observed_vals_b = Yb[:, t][Mb[:, t] == 1]
        if len(observed_vals_b) > 0:
            mean_t = observed_vals_b.mean()
            std_t = observed_vals_b.std()
            if std_t < 1e-6:
                std_t = 1.0
            Y_mean_b[t] = mean_t
            Y_stddev_b[t] = std_t
            Y_std_b[:, t] = (Yb[:, t] - mean_t) / std_t
        else:
            Y_std_b[:, t] = Yb[:, t]

    X_masked_std_b = np.zeros_like(Yb_fill)
    for t in range(T_periods):
        X_masked_std_b[:, t] = (Yb_fill[:, t] - Y_mean_b[t]) / Y_stddev_b[t]

    X_masked_std_b[Mb == 0] = 0.0

    X_std_b = np.hstack([X_masked_std_b, Mb, covariates_matrix_b])
    X_std_b = X_std_b.astype(np.float32)
    Y_std_b = Y_std_b.astype(np.float32)
    Mb = Mb.astype(np.float32)

    autoencoder_b = build_autoencoder(input_dim, output_dim)
    optimizer_b = Adam(learning_rate=0.001)

    best_loss_b = np.inf
    wait_b = 0
    best_weights_b = None

    for epoch in range(epochs):
        indices_b = np.random.permutation(X_std_b.shape[0])
        epoch_loss_b = 0.0
        n_batches_b = 0

        for start in range(0, X_std_b.shape[0], batch_size):
            end = min(start + batch_size, X_std_b.shape[0])
            batch_idx_b = indices_b[start:end]
            X_batch_b = X_std_b[batch_idx_b]
            Y_batch_b = Y_std_b[batch_idx_b]
            M_batch_b = Mb[batch_idx_b]

            with tf.GradientTape() as tape:
                Y_pred_b = autoencoder_b(X_batch_b, training=True)
                loss_b = masked_mse(Y_batch_b, Y_pred_b, M_batch_b)

            grads_b = tape.gradient(loss_b, autoencoder_b.trainable_variables)
            optimizer_b.apply_gradients(zip(grads_b, autoencoder_b.trainable_variables))
            epoch_loss_b += loss_b.numpy()
            n_batches_b += 1

        avg_loss_b = epoch_loss_b / n_batches_b

        if avg_loss_b < best_loss_b:
            best_loss_b = avg_loss_b
            wait_b = 0
            best_weights_b = autoencoder_b.get_weights()
        else:
            wait_b += 1
            if wait_b >= patience:
                break

    if best_weights_b is not None:
        autoencoder_b.set_weights(best_weights_b)

    Y_pred_std_b = autoencoder_b.predict(X_std_b, verbose=0)
    Y_pred_b = np.zeros_like(Y_pred_std_b)
    for t in range(T_periods):
        Y_pred_b[:, t] = Y_pred_std_b[:, t] * Y_stddev_b[t] + Y_mean_b[t]

    tau_rows = []
    for i in range(len(boot_ids)):
        for t_idx in range(T_periods):
            if Mb[i, t_idx] == 0:
                tau_rows.append({
                    'tau': Yb[i, t_idx] - Y_pred_b[i, t_idx],
                    'region': b_meta.loc[i, 'region'],
                    'digital_index': b_meta.loc[i, 'digital_index']
                })

    tau_df_b = pd.DataFrame(tau_rows)
    if len(tau_df_b) == 0:
        continue

    boot_overall.append(tau_df_b['tau'].mean())

    for reg in boot_region.keys():
        reg_vals = tau_df_b[tau_df_b['region'] == reg]['tau']
        boot_region[reg].append(reg_vals.mean() if len(reg_vals) > 0 else np.nan)

    high_vals = tau_df_b[tau_df_b['digital_index'] > med_digital]['tau']
    low_vals = tau_df_b[tau_df_b['digital_index'] <= med_digital]['tau']
    boot_high_digital.append(high_vals.mean() if len(high_vals) > 0 else np.nan)
    boot_low_digital.append(low_vals.mean() if len(low_vals) > 0 else np.nan)

    if (b + 1) % 10 == 0:
        print(f"Bootstrap replication {b+1}/{B} completed")

boot_overall = [x for x in boot_overall if not np.isnan(x)]
for reg in boot_region:
    boot_region[reg] = [x for x in boot_region[reg] if not np.isnan(x)]
boot_high_digital = [x for x in boot_high_digital if not np.isnan(x)]
boot_low_digital = [x for x in boot_low_digital if not np.isnan(x)]

ci_overall = np.percentile(boot_overall, [2.5, 97.5])
ci_high = np.percentile(boot_high_digital, [2.5, 97.5]) if boot_high_digital else [np.nan, np.nan]
ci_low = np.percentile(boot_low_digital, [2.5, 97.5]) if boot_low_digital else [np.nan, np.nan]
ci_region = {r: np.percentile(vals, [2.5, 97.5]) for r, vals in boot_region.items() if len(vals) > 0}

print("\nBootstrap 95% Confidence Intervals (B=50):")
print(f"Overall ATT: {ci_overall}")
print(f"High digital ATT: {ci_high}")
print(f"Low digital ATT: {ci_low}")
for r, ci in ci_region.items():
    print(f"Region {r}: {ci}")

# -----------------------------------
# -----------------------------------------------------------------------------
# 13. Required plots
# -----------------------------------------------------------------------------
plot_df = df.groupby(['time', 'ever_treated'])['outcome'].mean().reset_index()
plt.figure(figsize=(10,6))
sns.lineplot(data=plot_df, x='time', y='outcome', hue='ever_treated', marker='o')
plt.title('Mean Observed Outcome over Time')
plt.xlabel('Time')
plt.ylabel('Mean Outcome')
plt.legend(title='Ever Treated')
plt.grid(True)
plt.savefig("/Users/nikoslamprou/Desktop/MScBusinessEcon/python/final.assignment/mean_observed_outcome.png")
plt.close()

treated_units = df[df['ever_treated']==1]['unit_id'].unique()
selected_units = treated_units[:4]
for unit in selected_units:
    unit_df = df[df['unit_id'] == unit].sort_values('time')
    treat_t = unit_df['treat_time'].iloc[0]
    plt.figure(figsize=(8,5))
    plt.plot(unit_df['time'], unit_df['outcome'], 'o-', label='Observed')
    plt.plot(unit_df['time'], unit_df['pred0'], 's--', label='Predicted No-Treatment')
    plt.axvline(x=treat_t, color='red', linestyle='--', label='Treatment Start')
    plt.title(f'Unit {unit}')
    plt.xlabel('Time')
    plt.ylabel('Outcome')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"/Users/nikoslamprou/Desktop/MScBusinessEcon/python/final.assignment/unit_{unit}_plot.png")
    plt.close()

event_att_df = event_att.reset_index()
event_att_df.columns = ['event_time', 'ATT']
plt.figure(figsize=(10,6))
plt.plot(event_att_df['event_time'], event_att_df['ATT'], marker='o', linestyle='-')
plt.axhline(y=0, color='gray', linestyle='--')
plt.title('Event-Time Average Treatment Effect on Treated (ATT)')
plt.xlabel('Time relative to treatment')
plt.ylabel('ATT')
plt.grid(True)
plt.savefig("/Users/nikoslamprou/Desktop/MScBusinessEcon/python/final.assignment/event_time_att.png")
plt.close()



##-----------------------------------------
# Export labels for Exercise 2
# -----------------------------------------------------------------------------
unit_tau_ex2 = df[df['treated_post']].groupby('unit_id')['tau'].mean().reset_index()
unit_tau_ex2.columns = ['unit_id', 'tau_i']
median_tau = unit_tau_ex2['tau_i'].median()
unit_tau_ex2['H_i'] = (unit_tau_ex2['tau_i'] > median_tau).astype(int)

unit_tau_ex2.to_csv("/Users/nikoslamprou/Desktop/MScBusinessEcon/python/final.assignment/unit_effect_labels.csv", index=False)

print("\n--- Exercise 1 completed successfully ---")
print("Plots and label file saved in:")
print("/Users/nikoslamprou/Desktop/MScBusinessEcon/python/final.assignment/")