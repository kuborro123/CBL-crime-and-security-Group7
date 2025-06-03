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


def allocate_resources(lsoas, ward_to_lsoas, lsoa_col):
    allocation_results = []

    for ward, lsoas_in_ward in ward_to_lsoas.items():
        # Collect raw risk scores for all LSOAs in this ward
        ward_lsoa_rows = lsoas[lsoas[lsoa_col].isin(lsoas_in_ward)]
        total_risk_in_ward = ward_lsoa_rows["risk_score"].sum()

        # If total risk is zero, assign zero officers/hours to every LSOA
        if total_risk_in_ward == 0:
            for lsoa in lsoas_in_ward:
                allocation_results.append({
                    "ward": ward,
                    "lsoa": lsoa,
                    "risk_score": 0,
                    "patrol_hours": 0,
                    "officers_allocated": 0
                })
            continue

        # Compute each LSOA's normalized risk and fractional officers
        frac_officers = {}
        for lsoa in lsoas_in_ward:
            risk_score = float(lsoas.loc[lsoas[lsoa_col] == lsoa, "risk_score"])
            normalized = risk_score / total_risk_in_ward
            frac_officers[lsoa] = normalized * MAX_OFFICERS_PER_WARD

        # Take floor of each fractional value and record the remainder
        floor_officers = {}
        remainders = {}
        for lsoa, frac_val in frac_officers.items():
            f = int(np.floor(frac_val))
            floor_officers[lsoa] = f
            remainders[lsoa] = frac_val - f

        # See how many officers remain to be assigned
        assigned_floor_sum = sum(floor_officers.values())
        remaining_officers = MAX_OFFICERS_PER_WARD - assigned_floor_sum

        # Distribute the remaining officers to the LSOAs with largest remainders
        sorted_by_remainder = sorted(
            remainders.items(),
            key=lambda x: x[1],
            reverse=True
        )
        extra_officer = {lsoa: 0 for lsoa in lsoas_in_ward}
        for i in range(remaining_officers):
            lsoa_to_boost = sorted_by_remainder[i][0]
            extra_officer[lsoa_to_boost] = 1

        # Build final integer allocations
        for lsoa in lsoas_in_ward:
            base = floor_officers[lsoa]
            extra = extra_officer[lsoa]
            officers_int = base + extra
            hours_int = officers_int * MAX_PATROL_HOURS_PER_OFFICER

            allocation_results.append({
                "ward": ward,
                "lsoa": lsoa,
                "risk_score": float(lsoas.loc[lsoas[lsoa_col] == lsoa, "risk_score"]),
                "patrol_hours": hours_int,
                "officers_allocated": officers_int
            })

    return pd.DataFrame(allocation_results)


# Define column names for LSOA and ward
lsoa_col = "lsoa21cd"
ward_col = "lad22nm"
wards = find_london_wards(WARDS_PATH)
lsoas, lsoa_col = load_lsoa_data(LSOA_SHAPE_PATH)

lsoa_predicted_risk_score = prediction_network()
lsoa_predicted_risk_score = lsoa_predicted_risk_score.rename(columns={"LSOA_code": "lsoa21cd",
                                                                      "predicted_value": "risk_score"})

lsoa_predicted_risk_score = lsoa_predicted_risk_score.set_index("lsoa21cd")["risk_score"].to_dict()

# Merge risk scores with LSOA data
lsoas["risk_score"] = lsoas["lsoa21cd"].map(lsoa_predicted_risk_score)
lsoas["risk_score"] = pd.to_numeric(lsoas["risk_score"], errors="coerce")
lsoas["risk_score"].fillna(0, inplace=True)

# Assign LSOA codes to wards and group lsoa by ward
lsoas_with_wards = gpd.sjoin(lsoas, wards, how="left", predicate="within")
ward_to_lsoas = lsoas_with_wards.groupby(ward_col)[lsoa_col].apply(list).to_dict()

allocation_df = allocate_resources(lsoas, ward_to_lsoas, lsoa_col)

# Output the allocation results
print("\n✅ Officer Allocation Results:")
print(allocation_df)

"""
np.random.seed(42)  # For reproducibility
mock_risk_scores = pd.DataFrame({
    "lsoa21cd": lsoas["lsoa21cd"],
    "risk_score": np.random.uniform(0.1, 1.0, size=len(lsoas))  # Random scores between 0.1 and 1.0
})

mock_risk_scores = mock_risk_scores.set_index("lsoa21cd")["risk_score"]

"""
