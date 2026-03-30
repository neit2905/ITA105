import pandas as pd
df = pd.read_csv('ITA105_Lab_3_Health.csv')

# Xem dữ liệu
print(df.head())

# Kiểm tra missing
print(df.isnull().sum())

# Thống kê
print(df.describe())
import matplotlib.pyplot as plt
import seaborn as sns

num_cols = df.select_dtypes(include=['int64','float64']).columns

# Histogram
df[num_cols].hist(figsize=(12,8))
plt.suptitle("Histogram - Patient Data")
plt.show()

# Boxplot
plt.figure(figsize=(12,6))
sns.boxplot(data=df[num_cols])
plt.title("Boxplot - Patient Data")
plt.show()
def detect_outliers(col):
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    return col[(col < lower) | (col > upper)]

for col in num_cols:
    outliers = detect_outliers(df[col])
    print(f"{col}: {len(outliers)} outliers")
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
    plt.title("MinMax")

    plt.subplot(1,3,3)
    sns.histplot(df_zscore[col], kde=True)
    plt.title("Z-score")

    plt.suptitle(col)
    plt.show()
for col in num_cols:
    plt.figure(figsize=(10,4))
    plt.boxplot([df[col], df_minmax[col], df_zscore[col]],
                labels=["Original","MinMax","Z-score"])
    plt.title(col)
    plt.show()
