#!/usr/bin/env python3
"""
police_allocation.py

Combined script for loading burglary data, computing risk by LSOA, and solving
an optimization model to allocate officer‐hours proportionally to burglary risk
with ward‐level constraints.
"""

import os
import glob
import zipfile
from collections import defaultdict

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pulp import LpProblem, LpVariable, lpSum, LpMaximize
import matplotlib.pyplot as plt

# Constants\ZIP_BURGLARY = "burglary_data.zip"  # Path to burglary CSV ZIP
EXTRACT_BURGLARY_DIR = "burglary_data"
ZIP_WARDS = "London-wards-2018.zip"  # Path to wards shapefile ZIP
EXTRACT_WARDS_DIR = "ward_shapefile_extracted"
ZIP_LSOA = "LB_LSOA2021_shp.zip"  # Path to LSOA shapefile ZIP
EXTRACT_LSOA_DIR = "LSOA_unzipped"
MAX_HOURS_PER_WARD = 800


def load_burglary_csvs(zip_path, extract_dir):
    # Unzip CSV files and find all .csv
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)
    return glob.glob(os.path.join(extract_dir, "**", "*.csv"), recursive=True)


def load_shapefile_from_zip(zip_path, extract_dir, rename_cols=None):
    # Unzip and locate .shp
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)
    shp_path = None
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f.lower().endswith('.shp'):
                shp_path = os.path.join(root, f)
                break
        if shp_path:
            break
    if not shp_path:
        raise FileNotFoundError(f"No .shp found in {extract_dir}")
    gdf = gpd.read_file(shp_path)
    if rename_cols:
        gdf = gdf.rename(columns=rename_cols)
    return gdf.to_crs(epsg=4326)


def load_lsoas(zip_path, extract_dir):
    # Unzip and load all LSOA shapefiles
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)
    gdfs = []
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f.lower().endswith('.shp'):
                try:
                    g = gpd.read_file(os.path.join(root, f)).to_crs(epsg=4326)
                    gdfs.append(g)
                except Exception as e:
                    print(f"Error reading {f}: {e}")
    return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")


def combine_burglary_data(csv_files):
    dfs = []
    for fn in csv_files:
        df = pd.read_csv(fn)
        if 'Crime type' in df.columns:
            df = df[df['Crime type'].str.lower() == 'burglary']
            if not df.empty:
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def spatial_join_burglaries(burglary_df, lsoas):
    # Drop missing coords
    df = burglary_df.dropna(subset=['Longitude', 'Latitude'])
    points = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=points, crs='EPSG:4326')
    # Clip to London
    london_union = lsoas.unary_union
    gdf = gdf[gdf.geometry.within(london_union)]
    # Spatial join
    lsoa_col = 'lsoa21cd'
    ward_col = 'lad22nm'
    joined = gpd.sjoin(
        gdf, lsoas[[lsoa_col, ward_col, 'geometry']],
        how='inner', predicate='within'
    )
    return joined, lsoa_col, ward_col


def compute_risk(joined, lsoa_col, ward_col):
    risk = (
        joined.groupby([lsoa_col, ward_col])
        .size()
        .reset_index(name='risk_score')
    )
    return risk


def solve_allocation(risk_df, max_hours):
    # Prepare data
    lsoas_list = risk_df[lsoa_col].tolist()
    risk_dict = dict(zip(risk_df[lsoa_col], risk_df['risk_score']))
    ward_map = dict(zip(risk_df[lsoa_col], risk_df[ward_col]))
    ward_to_lsoas = defaultdict(list)
    for lsoa, w in ward_map.items():
        ward_to_lsoas[w].append(lsoa)
    # Normalize
    total = sum(risk_dict.values())
    norm = {l: v / total for l, v in risk_dict.items()}
    # LP
    prob = LpProblem('BurglaryAllocation', LpMaximize)
    x = {l: LpVariable(f'x_{l}', lowBound=0) for l in lsoas_list}
    prob += lpSum(norm[l] * x[l] for l in lsoas_list)
    for ward, llist in ward_to_lsoas.items():
        prob += lpSum(x[l] for l in llist) <= max_hours
    prob.solve()
    allocation = {l: x[l].value() for l in lsoas_list if x[l].value() > 0}
    return allocation


if __name__ == '__main__':
    # Load data
    csv_files = load_burglary_csvs(ZIP_BURGLARY, EXTRACT_BURGLARY_DIR)
    wards = load_shapefile_from_zip(ZIP_WARDS, EXTRACT_WARDS_DIR, rename_cols={'GSS_CODE':'ward_code','NAME':'ward_name'})
    lsoas = load_lsoas(ZIP_LSOA, EXTRACT_LSOA_DIR)
    # Combine burglary records
    df_burglary = combine_burglary_data(csv_files)
    if df_burglary.empty:
        print('No burglary data found.')
        exit(1)
    # Spatial join
    joined, lsoa_col, ward_col = spatial_join_burglaries(df_burglary, lsoas)
    # Compute risk
    risk_df = compute_risk(joined, lsoa_col, ward_col)
    print(f'Total risk entries: {len(risk_df)}')
    # Solve
    allocation = solve_allocation(risk_df, MAX_HOURS_PER_WARD)
    print('Optimal allocation:')
    for l, hrs in allocation.items():
        print(f'{l}: {hrs:.1f} hours')
    # Plot
    fig, ax = plt.subplots(figsize=(10,10))
    lsoas.plot(ax=ax, facecolor='none', edgecolor='gray', linewidth=0.5)
    # recreate points for plotting
    joined.plot(ax=ax, color='red', markersize=1, alpha=0.5)
    plt.title('Burglary Locations vs LSOA Boundaries')
    plt.show()
