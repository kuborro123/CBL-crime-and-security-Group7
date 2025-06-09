import geopandas as gpd
import pandas as pd

# Load your full GeoJSON
full_geo = gpd.read_file("Streamlit_files/london_lsoa.geojson")

# Load your dataset (adjust filename if needed)
crime_data = pd.read_csv("Streamlit_files/crimes_per_month_per_LSOA.csv")

# Get unique LSOA codes from your data
lsoa_codes = crime_data["LSOA_code"].unique()

# Filter GeoJSON to only those codes
filtered_geo = full_geo[full_geo["LSOA21CD"].isin(lsoa_codes)]

# Save the filtered GeoJSON
filtered_geo.to_file("Streamlit_files/london_lsoa_filtered.geojson", driver="GeoJSON")
