import Data_loader as dl
import Dataset_maker as dm
import pandas as pd
import glob
import zipfile
import os
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from pulp import *
from collections import defaultdict
from Dataset_maker import get_all_burglary_data

# Constants
WARDS_PATH = 'data/London-wards-2018'
LSOA_SHAPE_PATH = 'data/LB_LSOA2021_shp'
MAX_HOURS_PER_WARD = 800


def find_london_wards(wards_path):
    shp_path = None
    for root, dirs, files in os.walk(wards_path):
        for file in files:
            if file.endswith(".shp"):
                shp_path = os.path.join(root, file)
                break
        if shp_path:
            break

    wards = gpd.read_file(shp_path)
    wards = wards.rename(columns={
        "GSS_CODE": "ward_code",
        "NAME": "ward_name"
    })
    wards = wards.to_crs(epsg=4326)
    return wards


def load_lsoa_data(lsoa_shape_path):
    lsoa_gdfs = []
    for root, dirs, files in os.walk(lsoa_shape_path):
        for file in files:
            if file.endswith(".shp"):
                full_path = os.path.join(root, file)
                try:
                    gdf = gpd.read_file(full_path).to_crs(epsg=4326)
                    lsoa_gdfs.append(gdf)
                except Exception as e:
                    print(f"Error reading {file}: {e}")

    lsoas = gpd.GeoDataFrame(pd.concat(lsoa_gdfs, ignore_index=True), crs="EPSG:4326")
    lsoa_col = "lsoa21cd" if "lsoa21cd" in lsoas.columns else lsoas.columns[0]
    return lsoas, lsoa_col


def plot_burglary_data(lsoas, gdf):
    """
    Plot burglary locations and LSOA boundaries.
    Args:
        lsoas (GeoDataFrame): GeoDataFrame containing LSOA boundaries.
        gdf (GeoDataFrame): GeoDataFrame containing burglary locations.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    lsoas.plot(ax=ax, facecolor='none', edgecolor='gray', linewidth=0.5)
    gdf.plot(ax=ax, color='red', markersize=1, alpha=0.5)
    plt.title("Burglary Locations vs LSOA Boundaries")
    plt.show()


def setup_linear_program(lsoa_list, normalized_risk, ward_to_lsoas, max_hours_per_ward):
    prob = LpProblem("BurglaryLSOAAllocation", LpMaximize)
    x = {lsoa: LpVariable(f"x_{lsoa}", lowBound=0) for lsoa in lsoa_list}
    prob += lpSum(normalized_risk[lsoa] * x[lsoa] for lsoa in lsoa_list)

    for ward, lsoas_in_ward in ward_to_lsoas.items():
        prob += lpSum(x[lsoa] for lsoa in lsoas_in_ward) <= max_hours_per_ward

    return prob, x


# Get the geo information of the wards and the lsoa's plus the burglary data
wards = find_london_wards(WARDS_PATH)
lsoas, lsoa_col = load_lsoa_data(LSOA_SHAPE_PATH)
df = get_all_burglary_data()
df = df.dropna(subset=["Longitude", "Latitude"])

# Convert the dataframe from the burglary data into a GeoDataFrame based on the longitude and latitude columns
geometry = [Point(xy) for xy in zip(df["Longitude"], df["Latitude"])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

london_union = lsoas.unary_union
gdf = gdf[gdf.geometry.within(london_union)]

# Column names for LSOA and ward
lsoa_col = "lsoa21cd"
ward_col = "lad22nm"

gdf_with_lsoa = gpd.sjoin(gdf, lsoas[[lsoa_col, ward_col, "geometry"]], how="inner", predicate="within")
risk_by_lsoa = gdf_with_lsoa.groupby([lsoa_col, ward_col]).size().reset_index(name="risk_score")

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

# Set up and solve the linear programming problem
prob, x = setup_linear_program(lsoa_list, normalized_risk, ward_to_lsoas, MAX_HOURS_PER_WARD)
prob.solve()

# Output the results
print("\n✅ Optimal officer-hour allocation by LSOA (per week):")
for lsoa in lsoa_list:
    hours = x[lsoa].value()
    if hours and hours > 0:
        print(f"- {lsoa} ({lsoa_to_ward[lsoa]}): {hours:.1f} hours (risk: {risk_dict[lsoa]})")

plot_burglary_data(lsoas, gdf)
