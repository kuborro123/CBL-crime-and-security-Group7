import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from Data_loader import data_loader  
import os
import matplotlib.cm as cm
from matplotlib.colors import Normalize

if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

query = "SELECT * FROM crimes " 
df_crimes = data_loader(query)

print("data load done")
print(f"data shape: {df_crimes.shape}")
print(f"data columns: {df_crimes.columns.tolist()}")
print(f"Preview of the first 5 rows of data:")
print(df_crimes.head())

# Data preprocessing
# Convert string longitude and latitude to floating point numbers
df_crimes['Longitude'] = pd.to_numeric(df_crimes['Longitude'], errors='coerce')
df_crimes['Latitude'] = pd.to_numeric(df_crimes['Latitude'], errors='coerce')

# Make sure the Month column is of date type
if 'Month' in df_crimes.columns:
    try:
        df_crimes['Month'] = pd.to_datetime(df_crimes['Month'], format='%Y-%m', errors='coerce')
    except:
        print("Unable to convert Month column to date type, the original string will be used")

# 1. Crime type distribution visualization
plt.figure(figsize=(14, 10))
crime_counts = df_crimes['Crime type'].value_counts()
sns.barplot(x=crime_counts.values, y=crime_counts.index, palette='viridis')
plt.title('Crime type distribution in London', fontsize=18)
plt.xlabel('amount', fontsize=14)
plt.ylabel('crime type', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('visualizations/crime_types.png')
plt.show()

# 2. Crime geographic distribution scatter plot
plt.figure(figsize=(14, 12))
valid_loc = df_crimes.dropna(subset=['Longitude', 'Latitude'])
plt.scatter(valid_loc['Longitude'], valid_loc['Latitude'], 
           alpha=0.4, c='darkblue', s=15, edgecolor='none')
plt.title('The geography of crime in London', fontsize=18)
plt.xlabel('longitude', fontsize=14)
plt.ylabel('latitude', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('visualizations/crime_map.png')
plt.show()

# 3. Monthly Crime Trends Chart
if 'Month' in df_crimes.columns:
    if pd.api.types.is_datetime64_dtype(df_crimes['Month']):
        plt.figure(figsize=(16, 8))
        monthly_counts = df_crimes.groupby(df_crimes['Month'].dt.to_period('M')).size()
        monthly_counts.index = monthly_counts.index.astype(str)  
        monthly_counts.plot(kind='line', marker='o', linewidth=2, markersize=8, color='darkblue')
        plt.title('London Monthly Crime Trends', fontsize=18)
        plt.xlabel('month', fontsize=14)
        plt.ylabel('crime amount', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('visualizations/monthly_trend.png')
        plt.show()

# 4. Number of crimes reported by each police department
plt.figure(figsize=(14, 8))
reported_counts = df_crimes['Reported by'].value_counts()
sns.barplot(x=reported_counts.values, y=reported_counts.index, palette='Blues_d')
plt.title('Number of crimes reported by each police department', fontsize=18)
plt.xlabel('amount', fontsize=14)
plt.ylabel('police department', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('visualizations/reported_by.png')
plt.show()

# 5. Crime outcome classification
plt.figure(figsize=(14, 10))
outcome_counts = df_crimes['Last outcome category'].value_counts()
# If there are too many result categories, only the first 10 will be displayed
if len(outcome_counts) > 10:
    outcome_counts = outcome_counts.head(10)

sns.barplot(x=outcome_counts.values, y=outcome_counts.index, palette='RdPu_r')
plt.title('Crime Result Classification (Top 10)', fontsize=18)
plt.xlabel('amount', fontsize=14)
plt.ylabel('outcome type', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('visualizations/outcome_categories.png')
plt.show()

# 6. Crime density in London(Heat map)
plt.figure(figsize=(14, 12))
valid_loc = df_crimes.dropna(subset=['Longitude', 'Latitude'])
hexbin = plt.hexbin(valid_loc['Longitude'], valid_loc['Latitude'], 
           gridsize=50, cmap='YlOrRd', mincnt=1)
cb = plt.colorbar(label='crime amount')
cb.ax.tick_params(labelsize=12)
plt.title('Crime density in London', fontsize=18)
plt.xlabel('longitude', fontsize=14)
plt.ylabel('latitude', fontsize=14)
plt.tight_layout()
plt.savefig('visualizations/crime_heatmap.png')
plt.show()

# 7. Crime amounts by LSOA area
plt.figure(figsize=(16, 10))
# Get the top 15 LSOA areas with the most crimes
top_lsoa = df_crimes['LSOA name'].value_counts().head(15)
sns.barplot(x=top_lsoa.values, y=top_lsoa.index, palette='viridis')
plt.title('Crime-prone areas (top 15 LSOAs)', fontsize=18)
plt.xlabel('crime amount', fontsize=14)
plt.ylabel('LSOA areas', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('visualizations/crime_by_lsoa.png')
plt.show()

# 8. Comparison of geographical distribution of different crime types
plt.figure(figsize=(16, 12))
# Get the top 6 most common crime types and set color
top_crimes = df_crimes['Crime type'].value_counts().head(6).index.tolist()
colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
crime_color_map = dict(zip(top_crimes, colors))

for crime, color in crime_color_map.items():
    crime_data = df_crimes[df_crimes['Crime type'] == crime]
    crime_data = crime_data.dropna(subset=['Longitude', 'Latitude'])
    plt.scatter(crime_data['Longitude'], crime_data['Latitude'], 
              alpha=0.6, c=color, label=crime, s=20, edgecolor='none')

plt.title('Comparison of geographical distribution of different crime types', fontsize=18)
plt.xlabel('longitude', fontsize=14)
plt.ylabel('latitude', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/crime_types_map.png')
plt.show()

print("Visualization analysis is complete! All charts have been saved to the 'visualizations' folder.")