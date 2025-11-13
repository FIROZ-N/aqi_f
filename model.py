import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import  ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import pickle

# --- 1. Load the Original Dataset ---
df = pd.read_csv("AQI_CLeaned_1.csv")
df.head()

# --- 2. Data Cleaning and Missing Value Handling ---
print("null", df.isna().sum())

# -----------------------------
# . Identify columns
# -----------------------------

# Categorical columns (change if needed)
cat_cols = ['City', 'Season','Station']

# Numeric columns (all pollutant levels)
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove target from numeric scaling
num_cols = [col for col in num_cols if col != 'AQI']

# -----------------------------
# 3. Label Encoding
# -----------------------------
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le   # store if needed for future prediction


# -----------------------------
# Final check
# -----------------------------
print("Encoding Completed")
print(df.head())


# -----------------------------
# Drop unwanted columns
# -----------------------------
df = df.drop([ 'PM25_Index', 'PM10_Index',
       'NO2_Index', 'SO2_Index', 'CO_Index', 'O3_Index',
       'AQI_Category', 'Humidity', 'WindSpeed', 'WindDirection',
       'Hour', 'Month', 'DayOfWeek'], axis=1)

#Main model file
# ------------------------------
# Load dataset
# ------------------------------
features = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3']

X = df[features]
y = df["AQI"]

# ------------------------------
# Time-based split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ------------------------------
# Scaling
# ------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# Train model
# ------------------------------
gb = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=42
)

gb.fit(X_train_scaled, y_train)

# ------------------------------
# Predictions
# ------------------------------
y_pred = gb.predict(X_test_scaled)

# ------------------------------
# Evaluation
# ------------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 FAST-Tuned Gradient Boosting Performance:")
print(f"RMSE: {rmse:.3f}")
print(f"MAE : {mae:.3f}")
print(f"R²  : {r2:.3f}")

# ------------------------------
# Save Scaler and Model as Pickle Files
# ------------------------------

# Save the scaler
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
print("✅ Scaler saved as 'scaler.pkl'")

# Save the model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(gb, model_file)
print("✅ Model saved as 'model.pkl'")

# Optional: Save feature names for future reference
feature_info = {
    'features': features,
    'feature_names': list(X.columns)
}

with open('feature_info.pkl', 'wb') as feature_file:
    pickle.dump(feature_info, feature_file)
print("✅ Feature info saved as 'feature_info.pkl'")

print("\n🎯 All files saved successfully!")