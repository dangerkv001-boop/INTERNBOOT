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





import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('train.csv', parse_dates=['date'])
df.set_index('date', inplace=True)

# Aggregate daily sales
daily_sales = df['sales'].resample('D').sum()

# Moving averages
daily_sales['MA_7'] = daily_sales.rolling(window=7).mean()
daily_sales['MA_30'] = daily_sales.rolling(window=30).mean()

# Plot results
plt.figure(figsize=(14,6))
plt.plot(daily_sales.index, daily_sales, label='Actual Sales')
plt.plot(daily_sales['MA_7'], label='7-Day Moving Average')
plt.plot(daily_sales['MA_30'], label='30-Day Moving Average')
plt.title('Sales Forecast Using Moving Averages')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()





import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv('train.csv', parse_dates=['date'])

# Feature engineering
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Select features and target
X = df[['year', 'month', 'day', 'onpromotion']]
y = df['sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

