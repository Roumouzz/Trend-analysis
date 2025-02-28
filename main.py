# Step 1: Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA


# Step 2: Load the Data
# Make sure your CSV file is in the same directory or provide the full path.
df = pd.read_csv('/Users/roumouz/Desktop/code/shopping_trends.csv')

# Step 3: Data Cleaning & Preprocessing
# Convert the 'date' column to datetime objects.
df['date'] = pd.to_datetime(df['date'])

# Remove duplicate records, if any.
df = df.drop_duplicates()

# Fill missing values using forward-fill method (adjust based on your dataset).
df = df.fillna(method='ffill')

# Step 4: Set the Date as Index and Aggregate Sales
# Set the 'date' column as the index.
df.set_index('date', inplace=True)

# Resample the data to a monthly frequency by summing up daily sales.
monthly_sales = df['sales'].resample('M').sum()

# Step 5: Exploratory Data Analysis (EDA)
# Plot the monthly sales to visualize the trend.
plt.figure(figsize=(10, 5))
plt.plot(monthly_sales, marker='o', linestyle='-', label='Monthly Sales')
plt.title('Monthly Sales Trends')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Step 6: Decompose the Time Series (Trend & Seasonality)
# Use seasonal_decompose to break down the time series into trend, seasonal, and residual components.
decomposition = seasonal_decompose(monthly_sales, model='additive')
decomposition.plot()
plt.show()

# Step 7: Build a Forecasting Model with ARIMA
# Here we choose an ARIMA(1,1,1) model (p, d, q) – you may refine these parameters.
model = ARIMA(monthly_sales, order=(1, 1, 1))
model_fit = model.fit()
# Forecast the next 12 months.
forecast = model_fit.forecast(steps=12)
print("Forecasted Sales for Next 12 Months:\n", forecast)

# Step 8: Prepare the Forecast DataFrame
# Create a date range for the forecasted period starting from the month after the last observed date.
forecast_dates = pd.date_range(start=monthly_sales.index[-1] + pd.offsets.MonthBegin(), periods=12, freq='M')

# Create a DataFrame for the forecasted sales.
forecast_df = pd.DataFrame({'date': forecast_dates, 'forecast': forecast})
forecast_df.set_index('date', inplace=True)

# Step 9: Combine Historical and Forecasted Sales
# Merge the historical monthly sales and the forecasted sales into one DataFrame.
combined_df = pd.concat([monthly_sales, forecast_df], axis=1)
combined_df.columns = ['historical_sales', 'forecasted_sales']
combined_df.reset_index(inplace=True)

# Optional: Plot combined results to compare historical and forecasted data.
plt.figure(figsize=(10, 5))
plt.plot(combined_df['date'], combined_df['historical_sales'], label='Historical Sales', marker='o')
plt.plot(combined_df['date'], combined_df['forecasted_sales'], label='Forecasted Sales', marker='x', linestyle='--')
plt.title('Historical and Forecasted Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Step 10: Export the Combined Data for Dashboard Use
# Save the combined DataFrame as a CSV file to use in Tableau or Power BI.
combined_df.to_csv('sales_forecast_data.csv', index=False)

print("Data exported successfully to 'sales_forecast_data.csv'")
