import pandas as pd
df = pd.read_csv('ITA105_Lab_3_Sports.csv')

# Xem 5 dòng đầu
print(df.head())

# Kiểm tra missing values
print("\nMissing values:")
print(df.isnull().sum())

# Tk mô tả
print("\nStatistics:")
print(df.describe())
import matplotlib.pyplot as plt
import seaborn as sns

# Chọn các cột số
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Histogram
df[num_cols].hist(figsize=(12, 8))
plt.suptitle("Histogram - Original Data")
plt.show()

# Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[num_cols])
plt.title("Boxplot - Original Data")
plt.show()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_minmax = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)

print(df_minmax.head())
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_zscore = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)

print(df_zscore.head())
for col in num_cols:
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    sns.histplot(df[col], kde=True)
    plt.title(f"{col} - Original")

    plt.subplot(1,3,2)
    sns.histplot(df_minmax[col], kde=True)
    plt.title(f"{col} - MinMax")

    plt.subplot(1,3,3)
    sns.histplot(df_zscore[col], kde=True)
    plt.title(f"{col} - Z-score")

    plt.tight_layout()
    plt.show()
for col in num_cols:
    plt.figure(figsize=(10,4))

    data = [df[col], df_minmax[col], df_zscore[col]]
    plt.boxplot(data, labels=["Original", "MinMax", "Z-score"])

    plt.title(f"Boxplot Comparison - {col}")
    plt.show()
    