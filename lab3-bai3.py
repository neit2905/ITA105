import pandas as pd
df = pd.read_csv('ITA105_Lab_3_Finance.csv')

print(df.head())
print(df.info())
print(df.describe())
import matplotlib.pyplot as plt
import seaborn as sns

num_cols = df.select_dtypes(include=['int64','float64']).columns

plt.figure(figsize=(12,6))
sns.boxplot(data=df[num_cols])
plt.title("Boxplot - Company Data")
plt.show()
from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()
df_minmax = pd.DataFrame(minmax.fit_transform(df[num_cols]), columns=num_cols)
from sklearn.preprocessing import StandardScaler

zscore = StandardScaler()
df_zscore = pd.DataFrame(zscore.fit_transform(df[num_cols]), columns=num_cols)
# Original
plt.scatter(df['Revenue'], df['Profit'])
plt.title("Original")
plt.xlabel("Revenue")
plt.ylabel("Profit")
plt.show()

# MinMax
plt.scatter(df_minmax['Revenue'], df_minmax['Profit'])
plt.title("Min-Max")
plt.xlabel("Revenue")
plt.ylabel("Profit")
plt.show()

# Z-score
plt.scatter(df_zscore['Revenue'], df_zscore['Profit'])
plt.title("Z-score")
plt.xlabel("Revenue")
plt.ylabel("Profit")
plt.show()