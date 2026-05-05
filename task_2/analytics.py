import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df1 = pd.read_csv("Unemployment in India.csv")
df2 = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")


df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()


df1['Date'] = pd.to_datetime(df1['Date'])
df2['Date'] = pd.to_datetime(df2['Date'])


plt.figure(figsize=(10,5))
sns.lineplot(data=df2, x='Date', y='Estimated Unemployment Rate (%)')
plt.title("Unemployment Rate Over Time (Covid Impact)")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate")
plt.xticks(rotation=45)
plt.savefig('unemployment_over_time.png')
plt.show()


plt.figure(figsize=(12,6))
sns.barplot(data=df2, x='Region', y='Estimated Unemployment Rate (%)')
plt.title("State-wise Unemployment Rate")
plt.xticks(rotation=90)
plt.savefig('state_wise_unemployment.png')
plt.show()


plt.figure(figsize=(6,4))
sns.barplot(data=df1, x='Area', y='Estimated Unemployment Rate (%)')
plt.title("Rural vs Urban Unemployment")
plt.savefig('rural_vs_urban.png')
plt.show()


top_states = df2.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(8,5))
sns.barplot(x=top_states.values, y=top_states.index)
plt.title("Top 10 States with Highest Unemployment")
plt.xlabel("Unemployment Rate")
plt.ylabel("State")
plt.savefig('top_10_states.png')
plt.show()