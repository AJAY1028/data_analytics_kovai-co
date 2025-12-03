import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- Configuration ---
file_name = "Daily_Public_Transport_Passenger_Journeys_by_Service_Type_20250603.csv"
journey_cols = ['Local Route', 'Light Rail', 'Peak Service', 'Rapid Route', 'School', 'Other']

# --- 1. Data Loading, Cleaning, and Imputation (Phase 1 Steps) ---
try:
    df = pd.read_csv(file_name)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df['Other'] = df['Other'].fillna(0).astype(int)
    df['Total Journeys'] = df[journey_cols].sum(axis=1)
    df['DayOfWeek'] = df['Date'].dt.day_name()

    # Anomaly Imputation (as performed in the previous step)
    ANOMALY_DATE = pd.to_datetime('2024-09-29')
    avg_sunday = df[(df['DayOfWeek'] == 'Sunday') & (df['Date'] != ANOMALY_DATE)]['Total Journeys'].mean()
    df.loc[df['Date'] == ANOMALY_DATE, 'Total Journeys'] = int(round(avg_sunday))
    df['Total Journeys'] = df['Total Journeys'].astype(int)

except FileNotFoundError:
    print(f"Error: File '{file_name}' not found.")
    exit()

print("--- Phase 2: SARIMA Forecasting ---")

# --- 2. Data Preparation: Resample to Monthly Totals ---
# SARIMA performs best on a regular time series. We use 'ME' (Month End).
ts_monthly = df.set_index('Date')['Total Journeys'].resample('ME').sum()

# --- 3. Model Training (SARIMA) ---

# Define the SARIMA Model Parameters (These are common starting values)
# Order(p, d, q): Non-seasonal components (e.g., Autoregression, Differencing, Moving Average)
# Seasonal_Order(P, D, Q, s): Seasonal components (s=12 for monthly data)
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 12) 

print(f"\nTraining SARIMA{SARIMA_ORDER}x{SARIMA_SEASONAL_ORDER}...")

# Fit the model
model = SARIMAX(
    ts_monthly,
    order=SARIMA_ORDER,
    seasonal_order=SARIMA_SEASONAL_ORDER,
    enforce_stationarity=False,
    enforce_invertibility=False
)
model_fit = model.fit(disp=False)

print("SARIMA Model Summary:")
print(model_fit.summary())

# --- 4. Forecasting ---

# Define the number of future months to forecast (e.g., 12 months)
FORECAST_STEPS = 12 

# Get the forecast results
forecast_results = model_fit.get_forecast(steps=FORECAST_STEPS)
forecast_mean = forecast_results.predicted_mean
confidence_intervals = forecast_results.conf_int()

# --- 5. Visualization ---

plt.figure(figsize=(14, 7))

# Plot historical data
ts_monthly.plot(label='Historical Total Journeys', color='darkblue')

# Plot forecast mean
forecast_mean.plot(label='12-Month Forecast', color='red', linestyle='--')

# Plot confidence intervals (the likely range of the forecast)
plt.fill_between(
    confidence_intervals.index,
    confidence_intervals.iloc[:, 0], # Lower bound
    confidence_intervals.iloc[:, 1], # Upper bound
    color='pink',
    alpha=0.5,
    label='95% Confidence Interval'
)

plt.title(f'SARIMA Forecast of Monthly Public Transport Journeys ({FORECAST_STEPS} Months)')
plt.xlabel('Date')
plt.ylabel('Total Journeys (Monthly Sum)')
plt.ticklabel_format(style='plain', axis='y')
plt.legend()
plt.tight_layout()

plt.savefig("sarima_forecast_phase2.png")
plt.close()

print("\nPhase 2 Complete. SARIMA forecast plot saved as sarima_forecast_phase2.png.")