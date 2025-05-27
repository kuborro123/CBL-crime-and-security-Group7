import glob
import zipfile
import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from pulp import *
from collections import defaultdict


# Unzip the file
with zipfile.ZipFile("/content/6fc4df964bdce6a513e3dabe49e4f16276169b3f.zip", 'r') as zip_ref:
    zip_ref.extractall("/content/burglary_data")

csv_files = glob.glob("/content/burglary_data/**/*.csv", recursive=True)
print(f"Found {len(csv_files)} CSV files")
print(csv_files[:3])  # preview a few

"""###Importing London Wards ZIP file"""


# --- 1. Define paths ---
zip_path = "/content/London-wards-2018.zip"  # Replace with actual path
extract_dir = "/tmp/ward_shapefile_extracted"

# --- 2. Unzip ---
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_dir)

# --- 3. Recursively find the .shp file ---
shp_path = None
for root, dirs, files in os.walk(extract_dir):
    for file in files:
        if file.endswith(".shp"):
            shp_path = os.path.join(root, file)
            break
    if shp_path:
        break

assert shp_path is not None, "No .shp file found in the extracted ZIP."

# --- 4. Load shapefile ---
wards = gpd.read_file(shp_path)

# --- 5. Rename columns if needed ---
wards = wards.rename(columns={
    "GSS_CODE": "ward_code",
    "NAME": "ward_name"
})

# Optional: ensure proper CRS
wards = wards.to_crs(epsg=4326)

print("✅ Wards shapefile loaded successfully.")

"""###Importing LSOAs files"""



# --- Step 1: Unzip main ZIP ---
zip_path = "/content/LB_LSOA2021_shp.zip"  # Update if needed
extracted_path = "/content/LSOA_unzipped"

os.makedirs(extracted_path, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_path)

# --- Step 2: Recursively search for and load all .shp files ---
lsoa_gdfs = []

for root, dirs, files in os.walk(extracted_path):
    for file in files:
        if file.endswith(".shp"):
            full_path = os.path.join(root, file)
            try:
                gdf = gpd.read_file(full_path).to_crs(epsg=4326)
                lsoa_gdfs.append(gdf)
            except Exception as e:
                print(f"❌ Error reading {file}: {e}")

# --- Step 3: Combine into a single GeoDataFrame ---
lsoas = gpd.GeoDataFrame(pd.concat(lsoa_gdfs, ignore_index=True), crs="EPSG:4326")

# --- Step 4: Set LSOA column name ---
lsoa_col = "lsoa21cd" if "lsoa21cd" in lsoas.columns else lsoas.columns[0]

print(f"✅ Loaded {len(lsoas)} LSOA polygons.")

df_list = []

for file in csv_files:
    df = pd.read_csv(file)

    if "Crime type" not in df.columns:
        # print(f"Skipping {file} — missing 'Crime type'")
        continue

    df_burglary = df[df["Crime type"].str.lower() == "burglary"]

    if not df_burglary.empty:
        df_list.append(df_burglary)

# Combine
if df_list:
    combined_burglary_df = pd.concat(df_list, ignore_index=True)
    # print(f"Combined burglary dataset shape: {combined_burglary_df.shape}")
else:
    print("❌ No burglary records found in any file")



# --- 1. Load combined burglary dataset ---
df = combined_burglary_df
df = df[df["Crime type"].str.lower() == "burglary"]
df = df.dropna(subset=["Longitude", "Latitude"])

# --- 2. Convert to GeoDataFrame ---
geometry = [Point(xy) for xy in zip(df["Longitude"], df["Latitude"])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# --- 3. Load LSOA shapefiles (assume `lsoas` already loaded and merged) ---
lsoas = lsoas.to_crs(epsg=4326)  # Ensure matching CRS

# --- 4. Filter burglary points to only those within Greater London ---
london_union = lsoas.unary_union
gdf = gdf[gdf.geometry.within(london_union)]

# --- 5. Spatial join to assign burglaries to LSOAs ---
lsoa_col = "lsoa21cd"
ward_col = "lad22nm"  # Local authority name (ward-level proxy)

gdf_with_lsoa = gpd.sjoin(gdf, lsoas[[lsoa_col, ward_col, "geometry"]], how="inner", predicate="within")

# --- 6. Compute burglary risk per LSOA ---
risk_by_lsoa = gdf_with_lsoa.groupby([lsoa_col, ward_col]).size().reset_index(name="risk_score")

# --- Debug output ---
print(f"✅ Total burglaries: {len(gdf)}")
print(f"✅ Matched to LSOAs: {len(gdf_with_lsoa)}")
print(f"✅ Unique LSOAs: {gdf_with_lsoa[lsoa_col].nunique()}")
print(f"✅ Total risk entries: {len(risk_by_lsoa)}")

# --- 7. Optimization model setup ---
MAX_HOURS_PER_WARD = 800

# Build mappings
lsoa_list = risk_by_lsoa[lsoa_col].tolist()
risk_dict = dict(zip(risk_by_lsoa[lsoa_col], risk_by_lsoa["risk_score"]))
lsoa_to_ward = dict(zip(risk_by_lsoa[lsoa_col], risk_by_lsoa[ward_col]))

# Group LSOAs by ward
ward_to_lsoas = defaultdict(list)
for lsoa, ward in lsoa_to_ward.items():
    ward_to_lsoas[ward].append(lsoa)

# Normalize burglary risk
total_risk = sum(risk_dict.values())
normalized_risk = {lsoa: risk_dict[lsoa] / total_risk for lsoa in lsoa_list}

# --- 8. Linear programming ---
prob = LpProblem("BurglaryLSOAAllocation", LpMaximize)

# Decision variables: officer-hours per LSOA
x = {lsoa: LpVariable(f"x_{lsoa}", lowBound=0) for lsoa in lsoa_list}

# Objective: maximize risk-weighted allocation
prob += lpSum(normalized_risk[lsoa] * x[lsoa] for lsoa in lsoa_list)

# Constraints: Max hours per ward
for ward, lsoas_in_ward in ward_to_lsoas.items():
    prob += lpSum(x[lsoa] for lsoa in lsoas_in_ward) <= MAX_HOURS_PER_WARD

# --- 9. Solve ---
prob.solve()

# --- 10. Output ---
print("\n✅ Optimal officer-hour allocation by LSOA (per week):")
for lsoa in lsoa_list:
    hours = x[lsoa].value()
    if hours and hours > 0:
        print(f"- {lsoa} ({lsoa_to_ward[lsoa]}): {hours:.1f} hours (risk: {risk_dict[lsoa]})")

print(lsoas.columns)
print("Wards represented:", gdf_with_lsoa[ward_col].unique())


# Plot all LSOAs
fig, ax = plt.subplots(figsize=(10, 10))
lsoas.plot(ax=ax, facecolor='none', edgecolor='gray', linewidth=0.5)

# Plot burglary points
gdf.plot(ax=ax, color='red', markersize=1, alpha=0.5)

plt.title("Burglary Locations vs LSOA Boundaries")
plt.show()
