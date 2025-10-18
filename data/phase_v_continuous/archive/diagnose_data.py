"""
Diagnostic script to check data types and shapes before training
"""
import pandas as pd
import numpy as np

# Check one session
session_path = 'src/data/continuous/20251017_125600_session'

# Load data
sensor_data = pd.read_csv(f'{session_path}/sensor_data.csv', skipinitialspace=True)
sensor_data.columns = sensor_data.columns.str.strip()

print("=" * 60)
print("SENSOR DATA DIAGNOSTICS")
print("=" * 60)
print(f"\nColumns: {list(sensor_data.columns)}")
print(f"\nData types:")
print(sensor_data.dtypes)
print(f"\nFirst few rows:")
print(sensor_data.head())

# Extract features
feature_cols = [col for col in sensor_data.columns if col not in ['timestamp', 'sensor']]
print(f"\n Feature columns: {feature_cols}")

features = sensor_data[feature_cols]
print(f"\nFeature data types:")
print(features.dtypes)

# Convert to numpy
features_np = features.values
print(f"\nNumPy array shape: {features_np.shape}")
print(f"NumPy array dtype: {features_np.dtype}")

# Check for non-numeric values
print(f"\nChecking for NaN values:")
print(features.isna().sum())

print(f"\nChecking for object columns in features:")
object_cols = features.select_dtypes(include=['object']).columns.tolist()
if object_cols:
    print(f"⚠️  WARNING: Object columns found: {object_cols}")
    for col in object_cols:
        print(f"\n  {col} sample values:")
        print(f"  {features[col].head()}")
else:
    print("✅ All feature columns are numeric")

# Try to convert to float explicitly
try:
    features_float = features.astype(np.float32)
    print(f"\n✅ Successfully converted to float32")
    print(f"   Shape: {features_float.shape}")
    print(f"   Dtype: {features_float.dtype}")
except Exception as e:
    print(f"\n❌ Error converting to float: {e}")
