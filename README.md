# INTERNBOOT
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
df = pd.read_csv('train.csv', parse_dates=['date'])
df.set_index('date', inplace=True)

# Quick Stats
print(df[['sales']].describe())
print(df[['sales']].median())

# Check missing values
print(df.isna().sum())

# Plot daily total sales
daily_sales = df.resample('D')['sales'].sum()
plt.figure(figsize=(14,5))
plt.plot(daily_sales, label='Daily Sales')
plt.title("Daily Total Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.show()

# Seasonality: monthly sales
monthly_sales = df.resample('M')['sales'].sum()
plt.figure(figsize=(14,5))
monthly_sales.plot(kind='bar')
plt.title("Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()
