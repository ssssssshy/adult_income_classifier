import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

processed_path = "../data/processed/adult_processed_v1_features_scaled.csv"

df = pd.read_csv(processed_path)

print(df.shape)
df.head()

class_counts = df["income"].value_counts()
class_percent = df["income"].value_counts(normalize=True) * 100

print("Абсолютные значения:")
print(class_counts)
print("\nПроценты:")
print(class_percent.round(2))

plt.figure(figsize=(6, 4))
sns.barplot(x=class_counts.index, y=class_counts.values, palette="pastel")
plt.title("Распределение классов в целевой переменной (income)")
plt.xticks([0, 1], ["<=50K", ">50K"])
plt.ylabel("Количество образцов")
plt.xlabel("Класс")
plt.tight_layout()
plt.show()
