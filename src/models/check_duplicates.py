import pandas as pd

df = pd.read_parquet("data/processed/features.parquet")

features = df.drop(columns=[
    "cooler_condition",
    "valve_condition",
    "pump_condition",
    "accumulator_condition",
    "stable_flag",
])

print(features.duplicated().sum())
