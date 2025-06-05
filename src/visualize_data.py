import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

data_path = Path("../data/processed/adult_balanced.csv")
df = pd.read_csv(data_path)

num_cols = df.columns.drop("income")

plots_per_row = 3
num_rows = (len(num_cols) + plots_per_row - 1) // plots_per_row

fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(18, 4 * num_rows))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.histplot(df[col], kde=True, ax=axes[i], bins=30)
    axes[i].set_title(f"Распределение {col}")
    axes[i].tick_params(axis="x", rotation=45)


for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 12))
corr = df.corr()

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    annot=False,
    fmt=".2f",
    square=True,
    cbar_kws={"shrink": 0.5},
)
plt.title("Корреляционная матрица признаков")
plt.show()
