import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path


df = pd.read_csv("../data/raw/adult.csv")

print(df.info())
print(df.isnull().sum())
print(df.head())

df.replace("?", np.nan, inplace=True)
print(df.isnull().sum())
df.dropna(inplace=True)

categorical_cols = [
    "workclass",
    "education",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native.country",
]

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

df["income"] = df["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)

num_cols = [
    "age",
    "fnlwgt",
    "education.num",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
]

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


print(df.head())
print(df.info())


processed_dir = Path("../data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

output_path = processed_dir / "adult_processed_v1_features_scaled.csv"

df.to_csv(output_path, index=False)
print(f"✔️ Processed dataset saved to: {output_path.resolve()}")
