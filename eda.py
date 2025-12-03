import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
file_name = "Daily_Public_Transport_Passenger_Journeys_by_Service_Type_20250603.csv"
sns.set_style("whitegrid")
journey_cols = ['Local Route', 'Light Rail', 'Peak Service', 'Rapid Route', 'School', 'Other']

# --- 1. Data Loading and Feature Engineering ---
try:
    df = pd.read_csv(file_name)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    
    # Fill NaN values in 'Other' column with 0
    df['Other'] = df['Other'].fillna(0).astype(int)
    
    # Calculate Total Journeys
    df['Total Journeys'] = df[journey_cols].sum(axis=1)

    # Extract temporal features needed for the heatmap
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['MonthName'] = df['Date'].dt.strftime('%b') 
    df['MonthNum'] = df['Date'].dt.month

except FileNotFoundError:
    print(f"Error: File '{file_name}' not found. Check the file path.")
    exit()

# --- 2. Create Heatmap Data ---

# Order days for the heatmap (essential for correct display)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Determine correct month order based on month number
month_order = df[['MonthNum', 'MonthName']].drop_duplicates().sort_values('MonthNum')['MonthName'].tolist()

# Create a pivot table: Average Total Journeys per DayOfWeek and Month
heatmap_data = df.pivot_table(
    values='Total Journeys',
    index='DayOfWeek',
    columns='MonthName',
    aggfunc='mean'
).reindex(day_order) # Apply DayOfWeek order
heatmap_data = heatmap_data[month_order] # Apply Month order

# --- 3. Visualization: Heatmap ---
print("--- Generating Heatmap of Weekly/Monthly Patterns ---")
plt.figure(figsize=(14, 8))
sns.heatmap(
    heatmap_data,
    cmap='viridis',
    fmt=',.0f', # Format numbers with thousands separator
    linewidths=.5,
    linecolor='lightgray',
    annot=True,
    cbar_kws={'label': 'Average Daily Journeys'}
)
plt.title('Average Daily Journeys: Day of Week vs. Month', fontsize=16)
plt.ylabel('Day of Week', fontsize=12)
plt.xlabel('Month', fontsize=12)
plt.yticks(rotation=0)
plt.tight_layout()

# Save the plot
plt.savefig("ridership_patterns.png")
plt.close()

