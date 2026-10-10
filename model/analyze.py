
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
data = pd.read_csv(BASE_DIR / "data" / "gym_cleaned.csv")

# Correlation between numeric features and Max_BPM
correlations = (
    data.select_dtypes(include="number")
    .corr()["Max_BPM"]
    .sort_values(ascending=False)
)

print("\nCorrelation with Max_BPM:")
print(correlations)

# Visualize relationships
correlations.drop("Max_BPM").sort_values().plot(
    kind="barh",
    figsize=(10, 7)
)

plt.title("Feature Correlation with Max_BPM")
plt.xlabel("Pearson Correlation")
plt.tight_layout()
plt.show()

# Examine target distribution
print("\nMax_BPM statistics:")
print(data["Max_BPM"].describe())

plt.figure(figsize=(8, 5))
plt.hist(data["Max_BPM"], bins=25, edgecolor="black")
plt.title("Distribution of Max_BPM")
plt.xlabel("Maximum Heart Rate (BPM)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
