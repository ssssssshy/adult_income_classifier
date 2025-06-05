import pandas as pd
from pathlib import Path
from imblearn.over_sampling import SMOTE

input_path = Path("../data/processed/adult_processed_v1_features_scaled.csv")

df = pd.read_csv(input_path)

X = df.drop("income", axis=1)
y = df["income"]

print(f"Форма признаков: {X.shape}")
print(f"Распределение классов:\n{y.value_counts(normalize=True) * 100}")


smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"Новая форма признаков: {X_resampled.shape}")
print("Новое распределение классов:")
print(y_resampled.value_counts(normalize=True) * 100)

df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
df_balanced["income"] = y_resampled

output_path = Path("../data/processed/adult_balanced.csv")
df_balanced.to_csv(output_path, index=False)
