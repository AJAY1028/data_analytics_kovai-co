import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set plotting style
sns.set_style("whitegrid")

# Load and clean data (assuming initial steps have been run)
file_name = "Daily_Public_Transport_Passenger_Journeys_by_Service_Type_20250603.csv"
try:
    df = pd.read_csv(file_name)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df['Other'] = df['Other'].fillna(0).astype(int)
    journey_cols = ['Local Route', 'Light Rail', 'Peak Service', 'Rapid Route', 'School', 'Other']
    df['Total Journeys'] = df[journey_cols].sum(axis=1)
    df['DayOfWeek'] = df['Date'].dt.day_name() # Needed for boxplot
except FileNotFoundError:
    print(f"Error: File '{file_name}' not found.")
    exit()


# Resample data by Month End ('ME') and sum the journeys for each service type
# Note: Retaining 'ME' as per user's provided code structure, despite potential warning/error.
df_monthly_composition = df.set_index('Date')[journey_cols].resample('ME').sum()


# --- Create Combined Subplots ---
fig, axes = plt.subplots(2, 1, figsize=(14, 14))
plt.suptitle('Public Transport Journey Analysis: Composition and Seasonality', fontsize=16)

# 1. Stacked Area Plot (Composition over Time) - Axes[0]
df_monthly_composition.plot(
    kind='area',
    stacked=True,
    ax=axes[0],
    title='1. Monthly Journey Composition by Service Type (Stacked Area)'
)
axes[0].set_ylabel('Total Journeys (Monthly)')
axes[0].set_xlabel('Date')
axes[0].ticklabel_format(style='plain', axis='y')
axes[0].legend(title='Service Type', loc='upper left', fontsize='small')


# 2. Box Plot (Daily Seasonality) - Axes[1]
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sns.boxplot(x='DayOfWeek', y='Total Journeys', data=df, order=day_order, ax=axes[1], palette='pastel')

axes[1].set_title('2. Total Journeys Distribution by Day of the Week (Box Plot)')
axes[1].set_xlabel('Day of Week')
axes[1].set_ylabel('Daily Total Journeys')


plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout for suptitle
plt.savefig("combined_composition_boxplot.png")
plt.close()

print("Combined plot (Stacked Area and Box Plot) generated and saved as combined_composition_boxplot.png.")