import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from Data_loader import data_loader  
import os
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
import contextily as ctx  
from matplotlib.animation import FuncAnimation
import math
import random

if not os.path.exists('../visualizations'):
    os.makedirs('../visualizations')

query = "SELECT * FROM crimes_complete "
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
crime_counts = df_crimes['Crime_type'].value_counts()
sns.barplot(x=crime_counts.values, y=crime_counts.index, palette='viridis')
plt.title('Crime type distribution in London', fontsize=18)
plt.xlabel('amount', fontsize=14)
plt.ylabel('crime_type', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('visualizations/crime_types.png')
plt.show()

# 2. Crime geographic distribution scatter plot with London map background
plt.figure(figsize=(14, 12))
valid_loc = df_crimes.dropna(subset=['Longitude', 'Latitude'])

# Check if the data is within the London area
# Approximate longitude and latitude range of London
london_bounds = [-0.5, 0.3, 51.3, 51.7]  
# [minimum longitude, maximum longitude, minimum latitude, maximum latitude]

london_data = valid_loc[
    (valid_loc['Longitude'] >= london_bounds[0]) & (valid_loc['Longitude'] <= london_bounds[1]) &
    (valid_loc['Latitude'] >= london_bounds[2]) & (valid_loc['Latitude'] <= london_bounds[3])
]

plt.scatter(london_data['Longitude'], london_data['Latitude'], 
           alpha=0.6, c='red', s=15, edgecolor='none')

ctx.add_basemap(plt.gca(), crs="EPSG:4326", source=ctx.providers.CartoDB.Positron, alpha=0.7)

plt.title('The geography of crime in London', fontsize=18)
plt.xlabel('longitude', fontsize=14)
plt.ylabel('latitude', fontsize=14)
plt.tight_layout()
plt.savefig('visualizations/crime_map_with_background.png', dpi=300)
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
reported_counts = df_crimes['Reported_by'].value_counts()
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
outcome_counts = df_crimes['Last_outcome_category'].value_counts()
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

# 6. Crime density in London(Heat map) with London map background
plt.figure(figsize=(14, 12))

london_data = valid_loc[
    (valid_loc['Longitude'] >= london_bounds[0]) & (valid_loc['Longitude'] <= london_bounds[1]) &
    (valid_loc['Latitude'] >= london_bounds[2]) & (valid_loc['Latitude'] <= london_bounds[3])
]

colors = [(0, 0, 0, 0), (1, 0.1, 0, 0.9)] 
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

heatmap, xedges, yedges = np.histogram2d(
    london_data['Longitude'], london_data['Latitude'], 
    bins=100, range=[
        [london_bounds[0], london_bounds[1]], 
        [london_bounds[2], london_bounds[3]]
    ]
)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='YlOrRd', alpha=0.7, aspect='auto')

ctx.add_basemap(plt.gca(), crs="EPSG:4326", source=ctx.providers.CartoDB.Positron, alpha=0.8)

plt.colorbar(label='crime amount')
plt.title('Crime density in London', fontsize=18)
plt.xlabel('longitude', fontsize=14)
plt.ylabel('latitude', fontsize=14)
plt.tight_layout()
plt.savefig('visualizations/crime_heatmap_with_background.png', dpi=300)
plt.show()

# 7. Crime amounts by LSOA area
plt.figure(figsize=(16, 10))
# Get the top 15 LSOA areas with the most crimes
top_lsoa = df_crimes['LSOA_name'].value_counts().head(15)
sns.barplot(x=top_lsoa.values, y=top_lsoa.index, palette='viridis')
plt.title('Crime-prone areas (top 15 LSOAs)', fontsize=18)
plt.xlabel('crime amount', fontsize=14)
plt.ylabel('LSOA areas', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('visualizations/crime_by_lsoa.png')
plt.show()

# 8. Comparison of geographical distribution of different crime types with London map background
plt.figure(figsize=(16, 12))
# Get the top 6 most common crime types and set color
top_crimes = df_crimes['Crime_type'].value_counts().head(6).index.tolist()
colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
crime_color_map = dict(zip(top_crimes, colors))

for crime, color in crime_color_map.items():
    crime_data = df_crimes[df_crimes['Crime_type'] == crime]
    crime_data = crime_data.dropna(subset=['Longitude', 'Latitude'])
    crime_london_data = crime_data[
        (crime_data['Longitude'] >= london_bounds[0]) & (crime_data['Longitude'] <= london_bounds[1]) &
        (crime_data['Latitude'] >= london_bounds[2]) & (crime_data['Latitude'] <= london_bounds[3])
    ]
    plt.scatter(crime_london_data['Longitude'], crime_london_data['Latitude'], 
              alpha=0.6, c=color, label=crime, s=20, edgecolor='none')

ctx.add_basemap(plt.gca(), crs="EPSG:4326", source=ctx.providers.CartoDB.Positron, alpha=0.7)

plt.title('Comparison of geographical distribution of different crime types', fontsize=18)
plt.xlabel('longitude', fontsize=14)
plt.ylabel('latitude', fontsize=14)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/crime_types_map_with_background.png', dpi=300)
plt.show()

# ==== Burglary Analysis ====
burglary_df = df_crimes[df_crimes['Crime_type'] == 'Burglary']
print(f"Number of burglary cases: {len(burglary_df)}")

# 9. Monthly trend analysis of burglary
plt.figure(figsize=(14, 8))
if 'Month' in burglary_df.columns:
    if pd.api.types.is_datetime64_dtype(burglary_df['Month']):
        monthly_burglary = burglary_df.groupby(burglary_df['Month'].dt.to_period('M')).size()
        monthly_burglary.index = monthly_burglary.index.astype(str)
    
    monthly_burglary.plot(kind='line', marker='o', linewidth=2, markersize=8, color='crimson')
    plt.title('Monthly Burglary Trends', fontsize=18)
    plt.xlabel('month', fontsize=14)
    plt.ylabel('burglary amount', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/burglary_monthly_trend.png')
    plt.show()

# 10. Geography distribution of burglary with London map background
plt.figure(figsize=(14, 12))
valid_burglary = burglary_df.dropna(subset=['Longitude', 'Latitude'])
burglary_london = valid_burglary[
    (valid_burglary['Longitude'] >= london_bounds[0]) & (valid_burglary['Longitude'] <= london_bounds[1]) &
    (valid_burglary['Latitude'] >= london_bounds[2]) & (valid_burglary['Latitude'] <= london_bounds[3])
]

plt.scatter(burglary_london['Longitude'], burglary_london['Latitude'], 
           alpha=0.7, c='darkred', s=20, edgecolor='none')


ctx.add_basemap(plt.gca(), crs="EPSG:4326", source=ctx.providers.CartoDB.Positron, alpha=0.7)

plt.title('Geographical Distribution of Burglary in London', fontsize=18)
plt.xlabel('longitude', fontsize=14)
plt.ylabel('latitude', fontsize=14)
plt.tight_layout()
plt.savefig('visualizations/burglary_map_with_background.png', dpi=300)
plt.show()

# 11. Burglary heatmap with London map background
#plt.figure(figsize=(14, 12))
fig, ax = plt.subplots(figsize=(14, 12))
#colors = [(0, 0, 0, 0), (1, 0.1, 0, 1)]  
#cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

heatmap, xedges, yedges = np.histogram2d(
    burglary_london['Longitude'], burglary_london['Latitude'], 
    bins=25, range=[
        [-0.25, 0.05], 
        [51.45, 51.60]
    ]
)
# Create a modified version for animation
animated_heatmap = heatmap.copy()
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
# Initial heatmap display
im = ax.imshow(animated_heatmap.T, extent=extent, origin='lower', cmap='Reds', alpha=0.7, aspect='auto')
plt.colorbar(im, ax=ax, label='burglary amount')

# Set titles
ax.set_title('Burglary Density in London', fontsize=18)
ax.set_xlabel('longitude', fontsize=14)
ax.set_ylabel('latitude', fontsize=14)

# Animation function with individual cell flashing
def update(frame):
    # Create a copy of the original heatmap
    animated_heatmap = heatmap.copy()
    
    # Only modify cells with actual data
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            if heatmap[i, j] > 0:
                # Random chance for each cell to flash
                if random.random() < 0.2:  # 20% chance for a cell to flash
                    # Enhance the value for flashing
                    animated_heatmap[i, j] = heatmap[i, j] * (1.5 + random.random())
    
    # Update the heatmap image
    im.set_array(animated_heatmap.T)
    
    return [im]

# Create the animation (100 frames)
anim = FuncAnimation(fig, update, frames=100, interval=100, blit=True)

# Save as GIF
anim.save('visualizations/burglary_heatmap_animated.gif', 
          writer='pillow', fps=10, dpi=150)

plt.tight_layout()
plt.close()
#plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='Reds', alpha=0.7, aspect='auto')

#ctx.add_basemap(plt.gca(), crs="EPSG:4326", source=ctx.providers.CartoDB.Positron, alpha=0.8)

#plt.colorbar(label='burglary amount')
#plt.title('Burglary Density in London', fontsize=18)
#plt.xlabel('longitude', fontsize=14)
#plt.ylabel('latitude', fontsize=14)
#plt.tight_layout()
#plt.savefig('visualizations/burglary_heatmap_with_background.png', dpi=300)
#plt.show()

# 12. Distribution of burglary cases in each LSOA area
plt.figure(figsize=(16, 10))
lsoa_burglary = burglary_df['LSOA_name'].value_counts().head(15)
sns.barplot(x=lsoa_burglary.values, y=lsoa_burglary.index, palette='Reds_r')
plt.title('Burglary-prone areas (top 15 LSOAs)', fontsize=18)
plt.xlabel('burglary amount', fontsize=14)
plt.ylabel('LSOA areas', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('visualizations/burglary_by_lsoa.png')
plt.show()

# 13. Percentage of burglary
plt.figure(figsize=(12, 8))

df_crimes['Crime Category'] = df_crimes['Crime_type'].apply(
    lambda x: 'Burglary' if x == 'Burglary' else 'Other Crimes')
category_counts = df_crimes['Crime Category'].value_counts()

plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', 
        startangle=90, colors=['crimson', 'lightblue'], explode=(0.05, 0))
plt.title('Burglary vs Other Crimes', fontsize=18)
plt.axis('equal') 
plt.tight_layout()
plt.savefig('visualizations/burglary_vs_other.png')
plt.show()

# 14. Classification of burglary outcome
plt.figure(figsize=(14, 8))
if 'Last outcome category' in burglary_df.columns:
    outcome_burglary = burglary_df['Last_outcome_category'].value_counts()
    if len(outcome_burglary) > 10:
        outcome_burglary = outcome_burglary.head(10)
    sns.barplot(x=outcome_burglary.values, y=outcome_burglary.index, palette='YlOrBr')
    plt.title('Burglary Case Outcomes (Top 10)', fontsize=18)
    plt.xlabel('amount', fontsize=14)
    plt.ylabel('outcome category', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualizations/burglary_outcomes.png')
    plt.show()


print("Visualization analysis is complete! All charts have been saved to the 'visualizations' folder.")
