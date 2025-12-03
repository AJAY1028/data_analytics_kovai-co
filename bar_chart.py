import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_style("whitegrid")

# Define file name
file_name = "Daily_Public_Transport_Passenger_Journeys_by_Service_Type_20250603.csv"
journey_cols = ['Local Route', 'Light Rail', 'Peak Service', 'Rapid Route', 'School', 'Other']

# --- 1. Data Loading and Cleaning ---
try:
    df = pd.read_csv(file_name)
    
    # Fill NaN values in 'Other' column with 0
    df['Other'] = df['Other'].fillna(0).astype(int)
    
    # Calculate Total Journeys (not strictly needed for the bar chart, but good practice)
    df['Total Journeys'] = df[journey_cols].sum(axis=1)

except FileNotFoundError:
    print(f"Error: File '{file_name}' not found. Check the file path.")
    exit()

# --- 2. Visualization: Bar Chart of Average Service Usage ---
print("--- Generating Bar Chart: Average Daily Journeys by Service Type ---")

# Calculate the average journeys for each service type and sort them
avg_journeys = df[journey_cols].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=avg_journeys.index, y=avg_journeys.values, palette="viridis")

plt.title('Average Daily Journeys by Service Type', fontsize=16)
plt.ylabel('Average Journeys')
plt.xlabel('Service Type')
plt.ticklabel_format(style='plain', axis='y') # Prevent scientific notation
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the plot
plt.savefig("bar_chart_average.png")
plt.close()

