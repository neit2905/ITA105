import pandas as pd
df = pd.read_csv('ITA105_Lab_3_Gaming.csv')

print(df.head())

# Kiểm tra missing values
print("\nMissing values:")
print(df.isnull().sum())

# Thống kê mô tả
print("\nStatistics:")
print(df.describe())
import matplotlib.pyplot as plt
import seaborn as sns

num_cols = df.select_dtypes(include=['int64','float64']).columns

# Histogram
df[num_cols].hist(figsize=(12,8))
plt.suptitle("Histogram - Player Data")
plt.show()

# Boxplot
plt.figure(figsize=(12,6))
sns.boxplot(data=df[num_cols])
plt.title("Boxplot - Player Data")
plt.show()
from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()
df_minmax = pd.DataFrame(minmax.fit_transform(df[num_cols]), columns=num_cols)
from sklearn.preprocessing import StandardScaler

zscore = StandardScaler()
df_zscore = pd.DataFrame(zscore.fit_transform(df[num_cols]), columns=num_cols)
for col in num_cols:
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    sns.histplot(df[col], kde=True)
    plt.title("Original")

    plt.subplot(1,3,2)
    sns.histplot(df_minmax[col], kde=True)
    plt.title("Min-Max")

    plt.subplot(1,3,3)
    sns.histplot(df_zscore[col], kde=True)
    plt.title("Z-score")

    plt.suptitle(col)
    plt.show()
