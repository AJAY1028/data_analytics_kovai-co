import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- Configuration ---
file_name = "Daily_Public_Transport_Passenger_Journeys_by_Service_Type_20250603.csv"
journey_cols = ['Local Route', 'Light Rail', 'Peak Service', 'Rapid Route', 'School', 'Other']
FORECAST_DAYS = 7

# --- 1. Data Loading and Cleaning ---
try:
    df = pd.read_csv(file_name)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    
    # Fill NaNs immediately (crucial for forecasting)
    for col in journey_cols:
        df[col] = df[col].fillna(0)
    
    # Set Date as index and sort
    df = df.set_index('Date').sort_index()

except FileNotFoundError:
    print(f"Error: File '{file_name}' not found.")
    exit()

print("--- Daily Forecast Generation ---")

# --- 2. Iterative Forecasting (Daily SARIMAX) ---

forecast_results = {}

# We will iterate through the columns and forecast 7 days ahead
for col in journey_cols:
    # Use simple (0,1,0)x(1,0,0,7) model for daily seasonality
    ts = df[col]
    
    # Drop the first 100 days to exclude the most volatile early data and ensure stationarity
    ts_train = ts.iloc[100:] 

    # Handle cases where the series might be all zeros
    if ts_train.empty or ts_train.sum() == 0:
        forecast = [0] * FORECAST_DAYS
    else:
        try:
            model = SARIMAX(
                ts_train,
                order=(0, 1, 0),
                seasonal_order=(1, 0, 0, 7), # Period 7 for daily seasonality
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            model_fit = model.fit(disp=False, maxiter=100)
            forecast_output = model_fit.get_forecast(steps=FORECAST_DAYS)
            forecast = forecast_output.predicted_mean.round().astype(int).tolist()
        except Exception as e:
            # Fallback to simple last known value if SARIMAX fails
            last_value = ts_train.iloc[-1]
            forecast = [last_value] * FORECAST_DAYS
            print(f"Warning: SARIMAX failed for {col}. Falling back to last value.")

    forecast_results[col] = forecast

# --- 3. Format Output ---
df_forecast = pd.DataFrame(forecast_results)

# Create future dates for index
last_date = df.index.max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq='D')
df_forecast['Date'] = future_dates
df_forecast['DayOfWeek'] = future_dates.day_name()
df_forecast = df_forecast[['Date', 'DayOfWeek'] + journey_cols]


# --- FINAL OUTPUT: Using to_string() to bypass dependency issue ---
print("\n--- 7-Day Forecast (Predicted Journeys) ---")
print(df_forecast.to_string(index=False))