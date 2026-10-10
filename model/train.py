
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# 1. LOAD CLEANED DATA
# ==========================================

# Get the main project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset is inside the data folder
DATA_PATH = BASE_DIR / "data" / "gym_cleaned.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Run the cleaning script first."
    )

data = pd.read_csv(DATA_PATH)

print("Dataset shape:", data.shape)

print("\nMissing values:")
print(data.isnull().sum())


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

TARGET = "Max_BPM"

X = data.drop(columns=[TARGET])
y = data[TARGET]

print("\nFeatures:", X.columns.tolist())
print("Target:", TARGET)


# ==========================================
# 3. IDENTIFY FEATURE TYPES
# ==========================================

numeric_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_cols = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

print("\nNumeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)


# ==========================================
# 4. PREPROCESSING
# ==========================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])


# ==========================================
# 5. SPLIT DATA
# ==========================================

# 60% training, 20% validation, 20% testing

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))


# ==========================================
# 6. CREATE RANDOM FOREST MODEL
# ==========================================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ))
])


# ==========================================
# 7. TRAIN MODEL
# ==========================================

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training completed!")


# ==========================================
# 8. EVALUATE ON VALIDATION DATA
# ==========================================

predictions = model.predict(X_val)

mae = mean_absolute_error(y_val, predictions)
mse = mean_squared_error(y_val, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_val, predictions)

print("\n========== RANDOM FOREST RESULTS ==========")
print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")


# ==========================================
# 9. COMPARE WITH BASELINE
# ==========================================

baseline = DummyRegressor(strategy="mean")

baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_val)

baseline_mae = mean_absolute_error(
    y_val, baseline_predictions
)

baseline_r2 = r2_score(
    y_val, baseline_predictions
)

print("\n========== BASELINE RESULTS ==========")
print(f"Baseline MAE: {baseline_mae:.4f}")
print(f"Baseline R2:  {baseline_r2:.4f}")


# ==========================================
# 10. DISPLAY SAMPLE PREDICTIONS
# ==========================================

results = pd.DataFrame({
    "Actual Max BPM": y_val.to_numpy(),
    "Predicted Max BPM": predictions
})

results["Absolute Error"] = (
    results["Actual Max BPM"] -
    results["Predicted Max BPM"]
).abs()

print("\n========== SAMPLE PREDICTIONS ==========")
print(results.head(10))


# ==========================================
# 11. FEATURE IMPORTANCE
# ==========================================

feature_names = (
    model.named_steps["preprocessor"]
    .get_feature_names_out()
)

importances = (
    model.named_steps["regressor"]
    .feature_importances_
)

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(feature_importance.head(15))


# ==========================================
# 12. FINISH
# ==========================================

print("\nModel training and validation completed!")
print("Test dataset reserved for final evaluation.")
