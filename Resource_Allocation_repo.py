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
import numpy as np
from time_series_prediction import prediction_network


# place for the import of the necessary function from the prediction model


# Constants
WARDS_PATH = 'data/London-wards-2018'
LSOA_SHAPE_PATH = 'data/LB_LSOA2021_shp'
MAX_OFFICERS_PER_WARD = 100
MAX_PATROL_HOURS_PER_OFFICER = 2
TOTAL_PATROL_HOURS_PER_WARD = MAX_OFFICERS_PER_WARD * MAX_PATROL_HOURS_PER_OFFICER
MAX_HOURS_PER_WARD = 200


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
    """
    Set up the linear programming problem to allocate officer hours to LSOAs based on burglary risk.
    """
    prob = LpProblem("BurglaryLSOAAllocation", LpMaximize)
    x = {lsoa: LpVariable(f"x_{lsoa}", lowBound=0) for lsoa in lsoa_list}
    prob += lpSum(normalized_risk[lsoa] * x[lsoa] for lsoa in lsoa_list)

    for ward, lsoas_in_ward in ward_to_lsoas.items():
        prob += lpSum(x[lsoa] for lsoa in lsoas_in_ward) <= max_hours_per_ward

    return prob, x


# Define column names for LSOA and ward
lsoa_col = "lsoa21cd"
ward_col = "lad22nm"
wards = find_london_wards(WARDS_PATH)
lsoas, lsoa_col = load_lsoa_data(LSOA_SHAPE_PATH)

lsoa_predicted_risk_score = prediction_network()
print(lsoa_predicted_risk_score)

"""
np.random.seed(42)  # For reproducibility
mock_risk_scores = pd.DataFrame({
    "lsoa21cd": lsoas["lsoa21cd"],
    "risk_score": np.random.uniform(0.1, 1.0, size=len(lsoas))  # Random scores between 0.1 and 1.0
})

mock_risk_scores = mock_risk_scores.set_index("lsoa21cd")["risk_score"]

"""


# Merge risk scores with LSOA data
lsoas["risk_score"] = lsoas["lsoa21cd"].map(lsoa_predicted_risk_score)

# Assign LSOA codes to wards and group lsoa by ward
lsoas_with_wards = gpd.sjoin(lsoas, wards, how="left", predicate="within")
ward_to_lsoas = lsoas_with_wards.groupby(ward_col)[lsoa_col].apply(list).to_dict()

# For each ward distribute patrol hours to LSOAs based on risk scores and proximity between LSOAs inside the ward
# Allocate officers to LSOAs in each ward
allocation_results = []

# Iterate over ward_to_lsoas dictionary
for ward, lsoas_in_ward in ward_to_lsoas.items():
    # Get the total risk score for the ward
    total_risk_in_ward = lsoas[lsoas[lsoa_col].isin(lsoas_in_ward)]["risk_score"].sum()

    # Allocate patrol hours proportionally to risk scores
    for lsoa in lsoas_in_ward:
        lsoa_row = lsoas[lsoas[lsoa_col] == lsoa].iloc[0]
        risk_score = lsoa_row["risk_score"]
        normalized_risk = risk_score / total_risk_in_ward if total_risk_in_ward > 0 else 0
        patrol_hours = normalized_risk * TOTAL_PATROL_HOURS_PER_WARD
        officers_allocated = patrol_hours / MAX_PATROL_HOURS_PER_OFFICER

        # Store the allocation result
        allocation_results.append({
            "ward": ward,
            "lsoa": lsoa,
            "risk_score": risk_score,
            "patrol_hours": round(patrol_hours, 2),
            "officers_allocated": int(officers_allocated)
        })

# Convert results to a DataFrame for easier analysis
allocation_df = pd.DataFrame(allocation_results)

# Output the allocation results
print("\n✅ Officer Allocation Results:")
print(allocation_df)
