import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Configuration ---
file_name = "Daily_Public_Transport_Passenger_Journeys_by_Service_Type_20250603.csv"
journey_cols = ['Local Route', 'Light Rail', 'Peak Service', 'Rapid Route', 'School', 'Other']
plot_cols = ['Total Journeys', 'Rapid Route', 'Local Route', 'Light Rail', 'School'] 

# --- 1. Data Loading and Cleaning ---
try:
    df = pd.read_csv(file_name)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    
    # Handle NaNs and calculate Total Journeys immediately
    df['Other'] = df['Other'].fillna(0)
    df['Total Journeys'] = df[journey_cols].sum(axis=1).astype(int)
    
except FileNotFoundError:
    print(f"Error: File '{file_name}' not found.")
    exit()

# --- 2. Generate Histograms ---

sns.set_style("whitegrid")
num_plots = len(plot_cols)
n_cols = 2
n_rows = int(np.ceil(num_plots / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(plot_cols):
    sns.histplot(
        df[col], 
        bins=30, 
        kde=True, 
        ax=axes[i], 
        color='darkblue', 
        edgecolor='black'
    )
    axes[i].set_title(f'Distribution of {col}', fontsize=12)
    axes[i].set_xlabel('Journey Count')
    axes[i].ticklabel_format(style='plain', axis='x') # Prevents scientific notation

# Hide any unused subplots
for j in range(num_plots, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle('Histograms of Daily Public Transport Journey Counts', fontsize=16, y=1.02)
plt.tight_layout()

plt.savefig("histograms_distribution.png")
plt.close()