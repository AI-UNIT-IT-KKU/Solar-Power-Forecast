
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from solar_forecasting import (
    resample_data,
    normalize_columns,
    detect_outliers_iqr as sf_detect_outliers_iqr,
    replace_outliers_with_nan,
    fill_missing_cross,
    compute_pearson_correlation,
    compute_mutual_information,
    select_top_features,
    chronological_split,
    train_xgboost,
    evaluate_predictions,
)

# Alias the imported IQR outlier detector so the notebook‑style code can
# refer to ``detect_outliers_iqr`` directly without defining a wrapper
# around it. This avoids an unnecessary function‑calling‑a‑function and
# preserves clean design.
detect_outliers_iqr = sf_detect_outliers_iqr
from solar_forecasting.lstm_models import (
    build_vanilla_lstm,
    build_stacked_lstm,
    build_bidirectional_lstm,
    build_cnn_lstm,
)

files = [
    "df1_new.pkl",
    "df2_new.pkl",
    "df3_new.pkl",
    "df4_new.pkl",
    "df5_new.pkl"
]
dfs_raw = [pd.read_pickle("Data/" + f) for f in files]

"""1) Verify Time Consistency

First, check if all datasets follow the same time frequency (e.g., every second or every minute).
If they’re not perfectly aligned or have missing timestamps, resample them to a unified scale using:
"""

for df in dfs_raw:
    df['Time'] = pd.to_datetime(df['Time'])
    diffs = df['Time'].diff().dropna()
    print(diffs.value_counts().head())

"""525136 sample recorde in every minute

2933 duplicates sample in ech file

lets resample to 1min
"""

for df in dfs_raw:
    print(df.head(10))

def resample_sanity_check(df, freq="1min", time_column="Time"):
    df_copy = df.copy()

    print(">>> Before")
    print("shape:", df_copy.shape)

    if time_column in df_copy.columns:
        df_copy[time_column] = pd.to_datetime(df_copy[time_column], errors="coerce", utc=False)
        bad_ts = df_copy[time_column].isna().sum()
        print("bad timestamps:", bad_ts)
        df_copy = df_copy.dropna(subset=[time_column])
        resampled = resample_data(df_copy, freq=freq, time_column=time_column)
    else:
        # fall back to treating the index as the datetime axis
        df_copy.index = pd.to_datetime(df_copy.index, errors="coerce")
        bad_ts = df_copy.index.isna().sum()
        print("bad timestamps:", bad_ts)
        df_copy = df_copy[~df_copy.index.isna()]
        resampled = resample_data(df_copy, freq=freq)

    deltas = resampled.index.to_series().diff().dropna()
    if not deltas.empty:
        print("\nTop index deltas (counts):")
        print(deltas.value_counts().head())

    return resampled

dfs_resampled = [resample_sanity_check(df, freq="1min") for df in dfs_raw]

"""new shape >> shape: (527039, 22) | freq: 1min

**2) Standardize Column Names**
"""

for df in dfs_raw: print(df.columns.tolist())

def normalize_cols(df):
    df_norm = normalize_columns(df.copy())
    df_norm.columns = (
        df_norm.columns
        .str.replace('\n', ' ', regex=False)           # remove line breaks
        .str.replace(r'\s+\(', '(', regex=True)        # remove space before '('
        .str.replace(r'\s+', ' ', regex=True)          # collapse multiple spaces
        .str.strip()                                   # trim start/end spaces
    )
    return df_norm

dfs_resampled = [normalize_cols(df) for df in dfs_resampled]
dfs_resampled = fill_missing_cross(dfs_resampled)

"""now we have same columns name in all files"""


df_main = dfs_resampled[0]

"""## **handle outliers**"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- outlier detectors
# We alias detect_outliers_iqr to the helper imported from solar_forecasting.
# The Z‑score based detector is defined once below.
def detect_outliers_zscore(series, threshold: float = 3.0) -> pd.Series:
    """Detect outliers in a numeric series using the Z‑score method.

    Args:
        series: Numeric data to inspect.
        threshold: Number of standard deviations to use as the cutoff.

    Returns:
        A boolean Series indicating which values are considered outliers.
    """
    valid = series.dropna()
    if valid.empty:
        return pd.Series(False, index=series.index)
    mean = valid.mean()
    std = valid.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(False, index=series.index)
    z = (series - mean) / std
    mask = z.abs() > threshold
    return mask.fillna(False)

# --- choose numeric columns, exclude helper flags
numeric_df = df_main.filter(regex=r'^(?!.*(_was_na|_outlier))').select_dtypes('number')

# --- per-column counts
outlier_rows = []
for col in numeric_df.columns:                      # iterate columns explicitly
    s = df_main[col]
    m_iqr = detect_outliers_iqr(s)
    m_z   = detect_outliers_zscore(s)
    outlier_rows.append({
        'column': col,
        'iqr_count': int(m_iqr.sum()),
        'z_count':   int(m_z.sum()),
        'iqr_pct':   round(m_iqr.mean() * 100, 2),
        'z_pct':     round(m_z.mean()   * 100, 2),
    })

outlier_df = pd.DataFrame(outlier_rows).set_index('column')

"""### Mark and replace outliers with NaN"""


for col in df_main.select_dtypes('number'):
    mask = detect_outliers_iqr(df_main[col])
    df_main.loc[mask, col] = np.nan
#main df

for df in dfs_resampled[1:]:
    for col in df.select_dtypes('number'):
        mask = detect_outliers_iqr(df[col])
        df.loc[mask, col] = np.nan


"""**Handle Missing Values**

If datasets represent the same sensor/location, and the correlation is high (>0.95) → use one dataset to fill missing values in another at the same timestamp.

If the correlation is low or unstable, use temporal methods instead:
ffill, bfill, or time-based interpolation.

Why:
Forward/backward filling preserves the time order — crucial for forecasting — while cross-filling leverages sensor redundancy safely.

## Handling Missing Values Across Sensor Files

Since all sensor datasets represent the same physical system and share similar patterns, missing values are handled using a correlation-driven strategy:

Cross-Filling (when correlation > 0.95):
If two sensors show a very high correlation, missing values in one dataset are filled using the corresponding values from another sensor at the same timestamps.
This leverages redundancy between sensors and preserves real physical behavior.

Temporal Filling (when correlation < 0.95):
For sensors with lower or unstable correlation, missing values are filled using time-based methods — ffill, bfill, or interpolate(method='time').
This ensures temporal continuity without distorting the signal.

Smoothing (optional):
A light rolling median can be applied afterward to smooth small spikes or interpolation noise.

This approach combines cross-sensor reliability with time-series consistency, ensuring that the filled data remains both realistic and forecasting-friendly.
"""

# --- Step 3: Handle Missing Values (Simple + Practical) ---

# 2) use cross-filling if other files have similar structure
for df in dfs_resampled[1:]:
    # align timestamps
    df = df.reindex(df_main.index)
    # fill only where df_main has NaN and donor has data
    for col in df_main.columns.intersection(df.columns):
        mask = df_main[col].isna() & df[col].notna()
        df_main.loc[mask, col] = df.loc[mask, col]

# 3) fill remaining gaps using time-based interpolation + fallback
df_main = (
    df_main.interpolate(method='time', limit_direction='both')
            .ffill()
            .bfill()
)

# --- choose numeric columns, exclude helper flags
numeric_df = df_main.filter(regex=r'^(?!.*(_was_na|_outlier))').select_dtypes('number')

# --- per-column counts
outlier_rows = []
for col in numeric_df.columns:                      # iterate columns explicitly
    s = df_main[col]
    m_iqr = detect_outliers_iqr(s)
    m_z   = detect_outliers_zscore(s)
    outlier_rows.append({
        'column': col,
        'iqr_count': int(m_iqr.sum()),
        'z_count':   int(m_z.sum()),
        'iqr_pct':   round(m_iqr.mean() * 100, 2),
        'z_pct':     round(m_z.mean()   * 100, 2),
    })

outlier_df = pd.DataFrame(outlier_rows).set_index('column')


"""**computes the Pearson correlation between df1[column] and dfX[column]**"""

#rename the target column
df_main = df_main.rename(columns={'Control-PPC - Active power(1m)': 'active_power'})

# --- drop non-numeric columns (just in case)
numeric_df = df_main.filter(regex='^(?!.*(_was_na|_outlier))').select_dtypes('number')

# --- compute Pearson correlations via shared helper
corr_scores = compute_pearson_correlation(numeric_df, target='active_power')
corr_signed = numeric_df.corrwith(numeric_df['active_power']).drop('active_power')

print("Top correlations with active_power:\n")
print(corr_signed.sort_values(ascending=False))

plt.figure(figsize=(6, 10))
sns.heatmap(
    corr_signed.sort_values(ascending=False).to_frame(name='pearson'),
    annot=True,
    cmap='coolwarm',
    center=0,
)
plt.title("Pearson Correlation with active_power")
plt.show()

# Show weakest correlations using absolute values
corr_sorted = corr_signed.abs().sort_values()
weakest = corr_sorted.head(20).index.tolist()
print("Columns with weakest correlation to active_power:\n", weakest)
print("\nActual correlation values for these columns:\n")
print(corr_signed.loc[weakest])

# --- mutual information using shared helper
mi_scores = compute_mutual_information(numeric_df, target='active_power')
mi_df = mi_scores.sort_values(ascending=False).rename("Information_Gain").to_frame().reset_index()
mi_df = mi_df.rename(columns={'index': 'Feature'})

# simplified feature scoring + quick visuals (keeps same concepts: Pearson, Spearman, Mutual Info)
target_col = 'active_power'
num = df_main.select_dtypes('number').copy()

if target_col not in num.columns:
    raise KeyError(f"Target '{target_col}' not found in numeric dataframe")

X = num.drop(columns=[target_col])
y = num[target_col]

pearson_signed = corr_signed.reindex(X.columns)
spearman = X.corrwith(y, method='spearman')
scores = pd.concat(
    [
        pearson_signed.rename('pearson'),
        spearman.rename('spearman'),
        mi_scores.reindex(X.columns).rename('mi'),
    ],
    axis=1,
)
scores['pearson_abs'] = scores['pearson'].abs()
scores['mi_norm'] = scores['mi'] / (scores['mi'].max() or 1.0)

# derive feature sets from shared helper
features_pcc, features_ig = select_top_features(
    corr_scores,
    mi_scores,
    n_corr=14,
    n_mi=12,
)

"""### Comparative Training Strategy

Since Pearson Correlation (PCC) and Information Gain (IG) capture distinct relationships —
linear vs nonlinear — two separate training pipelines were established:

1. **PCC-based models:** trained only on features with strong linear correlation to `active_power`.
2. **IG-based models:** trained on features with high mutual information (nonlinear dependencies).

This dual-model approach allows direct comparison of how linear vs nonlinear features
affect forecasting accuracy across both XGBoost and LSTM architectures.

"""

# feature sets have already been derived using select_top_features above

TARGET = "active_power"



"""# XGBoost"""

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# 1) Build feature matrices from a cleaned copy
df_model = replace_outliers_with_nan(df_main)
df_model = df_model.ffill().bfill()

X_pcc = df_model[features_pcc].copy()
X_ig  = df_model[features_ig].copy()
y     = df_model[TARGET].copy()

# 2) Scale features (keeps target unscaled)
scaler_pcc = StandardScaler()
scaler_ig  = StandardScaler()
X_pcc_s = pd.DataFrame(scaler_pcc.fit_transform(X_pcc), index=X_pcc.index, columns=X_pcc.columns)
X_ig_s  = pd.DataFrame(scaler_ig.fit_transform(X_ig),   index=X_ig.index,   columns=X_ig.columns)


# 3) Chronological split (e.g., last 20% = test)
test_fraction = 0.20
Xp_tr, Xp_te, yp_tr, yp_te = chronological_split(X_pcc_s, y, test_fraction=test_fraction)
Xi_tr, Xi_te, yi_tr, yi_te = chronological_split(X_ig_s, y, test_fraction=test_fraction)


xgb_params = {
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "tree_method": "hist",
}

# 5) Train both models using shared helper
booster_pcc, pred_pcc, metrics_pcc = train_xgboost(
    Xp_tr,
    yp_tr,
    Xp_te,
    yp_te,
    label="XGB (PCC features)",
    params=xgb_params,
    num_boost_round=2000,
    early_stopping_rounds=100,
)

booster_ig, pred_ig, metrics_ig = train_xgboost(
    Xi_tr,
    yi_tr,
    Xi_te,
    yi_te,
    label="XGB (IG features)",
    params=xgb_params,
    num_boost_round=2000,
    early_stopping_rounds=100,
)

print("\n--- XGB (PCC features) ---")
print(metrics_pcc)
print("\n--- XGB (IG features) ---")
print(metrics_ig)

split_idx = len(y) - len(yp_te)

import numpy as np
import pandas as pd

# y_test & predictions (pred_pcc, pred_ig) already computed above
y_true = yp_te
pred_pcc_series = pd.Series(pred_pcc, index=Xp_te.index)
pred_ig_series  = pd.Series(pred_ig,  index=Xi_te.index)

# helper metrics
def mape(y, yhat, eps=1e-6):
    y = y.astype(float)
    return np.mean(np.abs((y - yhat) / np.maximum(np.abs(y), eps))) * 100

def smape(y, yhat, eps=1e-6):
    num = np.abs(yhat - y)
    den = (np.abs(y) + np.abs(yhat)) / 2.0
    return np.mean(num / np.maximum(den, eps)) * 100

def nrmse(y, yhat):
    rmse = np.sqrt(np.mean((y - yhat)**2))
    return {
        "by_range_%": rmse / (y.max() - y.min()) * 100,
        "by_std_%":   rmse / (y.std(ddof=0) + 1e-9) * 100
    }

# define daytime mask (use irradiance if available; fallback to power threshold)
if 'Total Solar Irradiance on Horizontal Plane GHI(W/m2)' in df_main.columns:
    ghi = df_main['Total Solar Irradiance on Horizontal Plane GHI(W/m2)'].reindex(y_true.index)
    day_mask = ghi > 50  # tweak as needed
else:
    day_mask = y_true > 1000  # fallback threshold

def summarize(y, yhat, tag):
    core_metrics = evaluate_predictions(y, yhat)
    res = {
        'MAE': core_metrics['mae'],
        'RMSE': core_metrics['rmse'],
        'R2': core_metrics['r2'],
        'MAPE_%': mape(y, yhat),
        'sMAPE_%': smape(y, yhat),
    }
    res.update({f'NRMSE_{k}': v for k, v in nrmse(y, yhat).items()})
    # daytime-only diagnostics
    y_d, yh_d = y[day_mask], yhat[day_mask]
    day_metrics = evaluate_predictions(y_d, yh_d)
    res['DAY_MAE'] = day_metrics['mae']
    res['DAY_RMSE'] = day_metrics['rmse']
    res['DAY_R2'] = day_metrics['r2']
    res['DAY_MAPE_%'] = mape(y_d, yh_d)
    res['DAY_sMAPE_%'] = smape(y_d, yh_d)
    return pd.Series(res, name=tag)

# baselines (same-time evaluation; simple references)
# 1) last-value baseline (1-step persistence)
baseline_last = y_true.shift(1).bfill()
# 2) yesterday-same-minute baseline (if test has >= 1440 mins)
daily_lag = 1440
baseline_yday = y_true.shift(daily_lag).bfill()

tbl = pd.concat([
    summarize(y_true, baseline_last, "Baseline_Last"),
    summarize(y_true, baseline_yday, "Baseline_Yesterday"),
    summarize(y_true, pred_pcc_series, "XGB_PCC"),
    summarize(y_true, pred_ig_series,  "XGB_IG"),
], axis=1)

print(tbl.round(3))

"""The results clearly show that XGBoost using Information Gain (IG) features achieved the best performance across all metrics.
Its lower MAE and RMSE values, along with a higher R², indicate that the model effectively captures both linear and nonlinear relationships in the solar data.
The daytime errors (DAY_MAPE ≈ 8–9%) confirm that the model predicts sunlight-driven fluctuations accurately, while the baselines perform poorly under changing conditions.
Overall, these results demonstrate a reliable and generalizable forecasting model suitable for real-world solar power prediction.
"""

#The XGBoost model trained on Information Gain (IG) features achieved the highest accuracy (R² = 0.984), showing excellent ability to capture nonlinear relationships in the solar system data. Compared to the PCC-based model, it produced lower error and more stable day-time predictions, which makes it suitable for deployment in real-time forecasting applications.**

booster_ig.save_model("xgb_ig_model.json")

booster_pcc.save_model("xgb_ig_model.json")







"""# Vanilla Multivariate Single-Step LSTM Model"""

import tensorflow as tf
from tensorflow import keras


# put this at the VERY top of your notebook, before importing keras
import os
os.environ["KERAS_BACKEND"] = "tensorflow"   # ensure TF backend for Keras 3

import keras
from keras import layers, models, callbacks, optimizers
import joblib

"""i just choose IG features for lstm model cause it gave better results in xgboost model"""

# ---- Config ----
TARGET = "active_power"
FEATS  = features_ig            # or features_pcc
WINDOW_SIZES = [30, 60, 120]
HORIZON = 1
VAL_FRAC = 0.10
TEST_FRAC = 0.20
BATCH_SIZE = 256
EPOCHS = 50
PATIENCE = 5
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ---- Split chronologically ----
df = df_main.sort_index().copy()
n = len(df)
test_cut = int(n * (1 - TEST_FRAC))
trainval, test = df.iloc[:test_cut], df.iloc[test_cut:]
val_cut = int(len(trainval) * (1 - VAL_FRAC))
train, val = trainval.iloc[:val_cut], trainval.iloc[val_cut:]

X_train, X_val, X_test = train[FEATS].values, val[FEATS].values, test[FEATS].values
y_train, y_val, y_test = train[TARGET].values.reshape(-1,1), val[TARGET].values.reshape(-1,1), test[TARGET].values.reshape(-1,1)

# ---- Scale features ----
x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)
X_train, X_val, X_test = x_scaler.transform(X_train), x_scaler.transform(X_val), x_scaler.transform(X_test)
y_train, y_val, y_test = y_scaler.transform(y_train), y_scaler.transform(y_val), y_scaler.transform(y_test)

def make_sequences(X, y, window, horizon=1):
    xs, ys = [], []
    for i in range(len(X) - window - horizon + 1):
        xs.append(X[i:i+window])
        ys.append(y[i+window+horizon-1, 0])
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)

def build_lstm(input_shape):
    model = build_vanilla_lstm(input_shape, units=64)
    model.compile(
        optimizer=optimizers.AdamW(),
        loss="mse",
        metrics=['mae', 'mape'])

    return model

def inverse_metrics(y_true_sc, y_pred_sc):
    yt = y_scaler.inverse_transform(y_true_sc.reshape(-1,1)).ravel()
    yp = y_scaler.inverse_transform(y_pred_sc.reshape(-1,1)).ravel()
    mae  = mean_absolute_error(yt, yp)
    rmse = mean_squared_error(yt, yp, squared=False)
    r2   = r2_score(yt, yp)
    return mae, rmse, r2, yt, yp

results = {}

for W in WINDOW_SIZES:
    Xtr, ytr = make_sequences(X_train, y_train, window=W, horizon=HORIZON)
    Xva, yva = make_sequences(X_val,   y_val,   window=W, horizon=HORIZON)
    Xte, yte = make_sequences(X_test,  y_test,  window=W, horizon=HORIZON)

    model = build_lstm((W, Xtr.shape[-1]))

    es = callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    rlrop = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1)


    # visible progress (loss, mae, mape)
    history = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es, rlrop],
        shuffle=False,
    )

    y_pred = model.predict(Xte, verbose=0).ravel()
    mae, rmse, r2, y_true_inv, y_pred_inv = inverse_metrics(yte, y_pred)

    results[W] = {"history": history, "metrics": (mae, rmse, r2), "y_true": y_true_inv, "y_pred": y_pred_inv}



    model.save(f"LSTM_IG{W}.keras")

# ---- Print summary ----
for W, r in results.items():
    mae, rmse, r2 = r["metrics"]
    print(f"[LSTM window={W}]  MAE: {mae:,.0f}   RMSE: {rmse:,.0f}   R²: {r2:.3f}")

"""**i'll stuck w 60 window size**  nah never mind"""

# [LSTM window=30]  MAE: 8,361   RMSE: 15,118   R²: 0.979
# [LSTM window=60]  MAE: 7,038   RMSE: 13,794   R²: 0.983
# [LSTM window=120]  MAE: 5,398   RMSE: 13,298   R²: 0.984





"""# Stacked LSTM"""

def build_Stackedlstm(input_shape):
    model = build_stacked_lstm(input_shape, units=128, num_layers=2)
    model.compile(
        optimizer=optimizers.AdamW(),
        loss="mse",
        metrics=['mae', 'mape'])

    return model

for W in WINDOW_SIZES:
    Xtr, ytr = make_sequences(X_train, y_train, window=W, horizon=HORIZON)
    Xva, yva = make_sequences(X_val,   y_val,   window=W, horizon=HORIZON)
    Xte, yte = make_sequences(X_test,  y_test,  window=W, horizon=HORIZON)

    model = build_Stackedlstm((W, Xtr.shape[-1]))

    es = callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    rlrop = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1)


    # visible progress (loss, mae, mape)
    history = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es, rlrop],
        shuffle=False,
    )

    y_pred = model.predict(Xte, verbose=0).ravel()
    mae, rmse, r2, y_true_inv, y_pred_inv = inverse_metrics(yte, y_pred)

    results[W] = {"history": history, "metrics": (mae, rmse, r2), "y_true": y_true_inv, "y_pred": y_pred_inv}

    model.save(f"StackedLSTM_IG{W}.keras")

# ---- Print summary ----
for W, r in results.items():
    mae, rmse, r2 = r["metrics"]
    print(f"[LSTM window={W}]  MAE: {mae:,.0f}   RMSE: {rmse:,.0f}   R²: {r2:.3f}")



"""# Bidirectional LSTM"""

def build_Bidirectionalstm(input_shape):
    model = build_bidirectional_lstm(input_shape, units=128)
    model.compile(
        optimizer=optimizers.AdamW(),
        loss="mse",
        metrics=['mae', 'mape'])

    return model

for W in WINDOW_SIZES:
    Xtr, ytr = make_sequences(X_train, y_train, window=W, horizon=HORIZON)
    Xva, yva = make_sequences(X_val,   y_val,   window=W, horizon=HORIZON)
    Xte, yte = make_sequences(X_test,  y_test,  window=W, horizon=HORIZON)

    model = build_Bidirectionalstm((W, Xtr.shape[-1]))

    es = callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    rlrop = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1)


    # visible progress (loss, mae, mape)
    history = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es, rlrop],
        shuffle=False,
    )

    y_pred = model.predict(Xte, verbose=0).ravel()
    mae, rmse, r2, y_true_inv, y_pred_inv = inverse_metrics(yte, y_pred)

    results[W] = {"history": history, "metrics": (mae, rmse, r2), "y_true": y_true_inv, "y_pred": y_pred_inv}

    model.save(f"BI_LSTM_IG{W}.keras")

# ---- Print summary ----
for W, r in results.items():
    mae, rmse, r2 = r["metrics"]
    print(f"[LSTM window={W}]  MAE: {mae:,.0f}   RMSE: {rmse:,.0f}   R²: {r2:.3f}")

"""# CNN → LSTM"""

def build_CNNlstm(input_shape):
    model = build_cnn_lstm(input_shape, conv_filters=64, kernel_size=3, lstm_units=64)
    model.compile(
        optimizer=optimizers.AdamW(),
        loss="mse",
        metrics=['mae', 'mape'])

    return model

for W in WINDOW_SIZES:
    Xtr, ytr = make_sequences(X_train, y_train, window=W, horizon=HORIZON)
    Xva, yva = make_sequences(X_val,   y_val,   window=W, horizon=HORIZON)
    Xte, yte = make_sequences(X_test,  y_test,  window=W, horizon=HORIZON)

    model = build_CNNlstm((W, Xtr.shape[-1]))

    es = callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    rlrop = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1)


    # visible progress (loss, mae, mape)
    history = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es, rlrop],
        shuffle=False,
    )

    y_pred = model.predict(Xte, verbose=0).ravel()
    mae, rmse, r2, y_true_inv, y_pred_inv = inverse_metrics(yte, y_pred)

    results[W] = {"history": history, "metrics": (mae, rmse, r2), "y_true": y_true_inv, "y_pred": y_pred_inv}

    model.save(f"CNN_LSTM_IG{W}.keras")

# ---- Print summary ----
for W, r in results.items():
    mae, rmse, r2 = r["metrics"]
    print(f"[LSTM window={W}]  MAE: {mae:,.0f}   RMSE: {rmse:,.0f}   R²: {r2:.3f}")
