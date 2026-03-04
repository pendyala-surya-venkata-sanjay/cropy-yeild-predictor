# Cleaned and rewritten Crop Yield Predictor
# Single-file script you can paste into a Jupyter cell or save as .py
# Features:
# - Single consistent dataframe flow
# - Missing value handling with SimpleImputer
# - Outlier capping (IQR) applied reliably
# - ColumnTransformer for numeric + categorical preprocessing
# - Train multiple regressors (Linear, RandomForest, GradientBoosting, XGBoost optional)
# - Unified evaluation table (RMSE, MAE, MSE, R2)
# - Predict function for single-row input

# ----------------------- Imports -----------------------
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Optional: XGBoost if installed
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# ----------------------- Configuration -----------------------
# Set your CSV path here (change if needed)
DATA_PATH = Path('crop_yeild_dataset1.csv')  # <- change as needed
RANDOM_STATE = 42
TARGET_COL = 'Yield_kg_ha'  # change to correct target column name in your dataset

# ----------------------- Utility functions -----------------------

def cap_iqr(df, cols):
    """Cap outliers at 1.5 * IQR for specified columns (inplace on a copy)."""
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        series = pd.to_numeric(df[c], errors='coerce')
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[c] = series.clip(lower=lower, upper=upper)
    return df

# ----------------------- Load data -----------------------
print('\nLoading data from:', DATA_PATH)
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data file not found at {DATA_PATH}. Update DATA_PATH variable.")

raw = pd.read_csv(DATA_PATH)
print('Original shape:', raw.shape)
print('Columns:', raw.columns.tolist())

# ----------------------- Basic cleaning & target detection -----------------------
# If target column name differs, try to auto-detect common variants
if TARGET_COL not in raw.columns:
    candidates = [c for c in raw.columns if 'yield' in c.lower() or 'Yield' in c]
    if len(candidates) > 0:
        TARGET_COL = candidates[0]
        print('Auto-detected target column:', TARGET_COL)
    else:
        raise ValueError('Target column not found. Set TARGET_COL to the correct column name.')

# Make a working copy
df = raw.copy()

# Standardize column names (optional): strip whitespace
df.columns = [c.strip() for c in df.columns]

# ----------------------- Drop/convert obviously useless columns -----------------------
# Drop columns that are fully empty or are IDs with unique values (heuristic)
n_unique = df.nunique(dropna=True)
cols_drop = [c for c in df.columns if df[c].isna().all()]
cols_drop += [c for c in df.columns if n_unique.get(c,0) == df.shape[0] and c.lower().startswith(('id','sr','sno'))]
cols_drop = list(set(cols_drop))
if cols_drop:
    print('Dropping columns (empty or likely IDs):', cols_drop)
    df.drop(columns=cols_drop, inplace=True)

# ----------------------- Make sure target is numeric -----------------------
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')

# ----------------------- Missing value handling -----------------------
# Simple strategy: numeric -> median, categorical -> most frequent
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# exclude target from features
if TARGET_COL in numeric_cols:
    numeric_cols.remove(TARGET_COL)
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print('Numeric features:', numeric_cols)
print('Categorical features:', categorical_cols)

# Apply imputers to a copy before outlier capping (we need numeric series for IQR)
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

# ----------------------- Outlier capping -----------------------
# IQR capping on numeric features
df = cap_iqr(df, numeric_cols + [TARGET_COL])

# ----------------------- Feature engineering (example) -----------------------
# This section depends on your dataset; adjust or extend as needed.
# Example: if you have rainfall, temp columns, create simple interactions
# if 'Rainfall' in df.columns and 'Temperature' in df.columns:
#     df['Rain_Temp'] = df['Rainfall'] * df['Temperature']

# ----------------------- Final feature matrix / label
# Drop any rows where target is still missing
before_drop = df.shape[0]
df = df.dropna(subset=[TARGET_COL])
after_drop = df.shape[0]
if before_drop != after_drop:
    print(f'Dropped {before_drop-after_drop} rows with missing target.')

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(float)

# Recompute feature lists in case imputers or transformations added/removed columns
numeric_features = [c for c in numeric_cols if c in X.columns]
categorical_features = [c for c in categorical_cols if c in X.columns]

# ----------------------- Preprocessing pipeline -----------------------
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop',
    sparse_threshold=0
)

# ----------------------- Train-test split -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# Fit preprocessor on training data
print('\nFitting preprocessor...')
preprocessor.fit(X_train)

# Transform datasets
X_train_t = preprocessor.transform(X_train)
X_test_t = preprocessor.transform(X_test)
print('Transformed feature shape (train):', X_train_t.shape)

# ----------------------- Models to train -----------------------
models = {}
models['LinearRegression'] = LinearRegression()
models['RandomForest'] = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
models['GradientBoosting'] = GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_STATE)
if _HAS_XGB:
    models['XGBoost'] = XGBRegressor(n_estimators=200, random_state=RANDOM_STATE, eval_metric='rmse')

# ----------------------- Train & evaluate -----------------------
results = []
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_t, y_train)
    preds = model.predict(X_test_t)

    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results.append({
        "Model": name,
        "RMSE": rmse,
        "MSE": mse,
        "MAE": mae,
        "R2": r2
    })


results_df = pd.DataFrame(results).sort_values('RMSE')
print('\nModel comparison:')
print(results_df.reset_index(drop=True))

# ----------------------- Cross-validated RMSE (optional) -----------------------
cv_summary = []
for name, model in models.items():
    try:
        scores = -1 * cross_val_score(model, preprocessor.transform(X), y, cv=5,
                                       scoring='neg_root_mean_squared_error')
        cv_summary.append({'Model': name, 'CV_RMSE_mean': scores.mean(), 'CV_RMSE_std': scores.std()})
    except Exception as e:
        cv_summary.append({'Model': name, 'CV_RMSE_mean': np.nan, 'CV_RMSE_std': np.nan})

cv_df = pd.DataFrame(cv_summary).sort_values('CV_RMSE_mean')
print('\nCross-validated RMSE (5-fold):')
print(cv_df.reset_index(drop=True))

# ----------------------- Save best model (optional) -----------------------
best_name = results_df.iloc[0]['Model']
best_model = models[best_name]
print(f'\nBest model by RMSE: {best_name}')
# ----------------------- Save best model to PKL -----------------------
import pickle

# Save both model + preprocessor + feature list
with open("best_crop_yield_model.pkl", "wb") as f:
    pickle.dump({
        "model": best_model,
        "preprocessor": preprocessor,
        "features": X.columns.tolist()
    }, f)

print("Saved best model to best_crop_yield_model.pkl")
import pickle
import pandas as pd

# Load the saved model
with open("best_crop_yield_model.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
preprocessor = saved["preprocessor"]
features = saved["features"]

def predict_from_pkl(input_dict):
    df = pd.DataFrame([input_dict])
    
    # Ensure all features exist
    for col in features:
        if col not in df:
            df[col] = None
    
    df = df[features]
    
    X_processed = preprocessor.transform(df)
    return model.predict(X_processed)[0]


# ----------------------- Prediction helper -----------------------

def predict_row(input_dict_or_df):
    """
    Accepts a single dict or single-row DataFrame with the same feature names as the training data (X).
    Returns predicted yield and model metadata.
    """
    single = pd.DataFrame([input_dict_or_df]) if isinstance(input_dict_or_df, dict) else input_dict_or_df.copy()
    # Ensure columns aligned; add missing columns with NaN
    for c in X.columns:
        if c not in single.columns:
            single[c] = np.nan
    single = single[X.columns]  # reorder
    # Impute + transform
    Xt = preprocessor.transform(single)
    pred = best_model.predict(Xt)
    return float(pred[0])

# Example usage (uncomment & edit with real values):
# sample = X_test.iloc[0].to_dict()
# print('Example predict ->', predict_row(sample))

# ----------------------- End -----------------------
print('\nCleaned pipeline built. Use predict_row(dict_or_df) to get a prediction.')
