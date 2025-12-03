import pandas as pd
import numpy as np

# --- Configuration ---
file_name = "Daily_Public_Transport_Passenger_Journeys_by_Service_Type_20250603.csv"
journey_cols = ['Local Route', 'Light Rail', 'Peak Service', 'Rapid Route', 'School', 'Other']

# --- 1. Data Loading and Cleaning ---
try:
    df = pd.read_csv(file_name)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df['Other'] = df['Other'].fillna(0).astype(int)
    df['Total Journeys'] = df[journey_cols].sum(axis=1)
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['Is_Weekend'] = df['DayOfWeek'].isin(['Saturday', 'Sunday'])
    df['Month'] = df['Date'].dt.month
    df['MonthName'] = df['Date'].dt.strftime('%B')

    # Impute the anomaly (as done in the previous step, rounded average for Sunday)
    ANOMALY_DATE = pd.to_datetime('2024-09-29')
    avg_sunday = df[(df['DayOfWeek'] == 'Sunday') & (df['Date'] != ANOMALY_DATE)]['Total Journeys'].mean()
    df.loc[df['Date'] == ANOMALY_DATE, 'Total Journeys'] = int(round(avg_sunday))
    df['Total Journeys'] = df['Total Journeys'].astype(int)

except FileNotFoundError:
    print(f"Error: File '{file_name}' not found.")
    exit()

print("--- Operational Strategy Quantification ---")

# --- 2. Weekend Optimization (3.1) ---

weekday_avg = df[~df['Is_Weekend']]['Total Journeys'].mean()
weekend_avg = df[df['Is_Weekend']]['Total Journeys'].mean()
weekend_drop_ratio = (1 - (weekend_avg / weekday_avg)) * 100

print("\n## 2. Weekend Optimization Data")
print("---------------------------------")
print(f"Average Weekday Journeys: {weekday_avg:,.0f}")
print(f"Average Weekend Journeys: {weekend_avg:,.0f}")
print(f"**Ridership Drop (Basis for Schedule Cuts): {weekend_drop_ratio:,.1f}%**")


# --- 3. Peak Period Staffing (3.2) and Target Marketing (3.3) ---

# Calculate Average Daily Journeys by Day of Week
day_avg = df.groupby('DayOfWeek')['Total Journeys'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Calculate Average Daily Journeys by Month
month_avg = df.groupby(['Month', 'MonthName'])['Total Journeys'].mean().reset_index().sort_values('Month')

# Identify the Busiest Day and Month (Peaks)
peak_day = day_avg.idxmax()
peak_day_avg = day_avg.max()
peak_month = month_avg.loc[month_avg['Total Journeys'].idxmax()]

# Identify the Quietest Day and Month (Troughs)
trough_day = day_avg.idxmin()
trough_day_avg = day_avg.min()
trough_month = month_avg.loc[month_avg['Total Journeys'].idxmin()]


print("\n## 3. Peak Period & Trough Data")
print("---------------------------------")

print("### Peak Staffing Data (Highest Usage for Staffing/Capacity)")
print(f"Busiest Day of Week: **{peak_day}** (Avg. Journeys: {peak_day_avg:,.0f})")
print(f"Busiest Month: **{peak_month['MonthName']}** (Avg. Daily Journeys: {peak_month['Total Journeys']:,.0f})")

print("\n### Target Marketing Data (Lowest Usage for Promotions)")
print(f"Quietest Day of Week: **{trough_day}** (Avg. Journeys: {trough_day_avg:,.0f})")
print(f"Quietest Month: **{trough_month['MonthName']}** (Avg. Daily Journeys: {trough_month['Total Journeys']:,.0f})")