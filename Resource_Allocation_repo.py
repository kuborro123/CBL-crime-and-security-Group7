import geopandas as gpd
import pandas as pd
from pulp import *
from pulp import LpProblem, LpVariable, LpInteger, lpSum, LpMaximize, PULP_CBC_CMD
from time_series_prediction import prediction_network
import matplotlib.pyplot as plt


# Constants
WARDS_PATH = 'data/London-wards-2018'
LSOA_SHAPE_PATH = 'data/LB_LSOA2021_shp'
MAX_OFFICERS_PER_WARD = 100
MAX_PATROL_HOURS_PER_OFFICER = 2
TOTAL_PATROL_HOURS_PER_WARD = MAX_OFFICERS_PER_WARD * MAX_PATROL_HOURS_PER_OFFICER  # 200
NEIGHBOR_WEIGHT_ALPHA = 0.5


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


def allocate_resources(lsoas: gpd.GeoDataFrame, ward_to_lsoas: dict, lsoa_col: str, ward_name_col: str) -> pd.DataFrame:


    allocation_results = []

    for ward, lsoas_in_ward in ward_to_lsoas.items():
        # 1) Subset to just these LSOAs
        ward_lsoas = lsoas[lsoas[lsoa_col].isin(lsoas_in_ward)].copy()

        # 2) Build a dict {lcode: risk_score}, forcing nonnegative
        raw_risk = {
            row[lsoa_col]: max(float(row["risk_score"]), 0.0)
            for _, row in ward_lsoas.iterrows()
        }

        total_risk = sum(raw_risk.values())
        if total_risk <= 0:
            # If risk is zero everywhere, allocate zero officers
            for lcode in lsoas_in_ward:
                allocation_results.append({
                    "ward": ward,
                    "lsoa": lcode,
                    "risk_score": 0.0,
                    "officers_allocated": 0,
                    "patrol_hours": 0,
                })
            continue

        # 3) Set up PuLP ILP: one integer var per LSOA in this ward
        prob = LpProblem(f"Allocate_{ward}", LpMaximize)
        y = {
            lcode: LpVariable(f"y_{ward}_{lcode}", lowBound=0, cat=LpInteger)
            for lcode in lsoas_in_ward
        }

        # 4) Constraint: total officers = MAX_OFFICERS_PER_WARD
        prob += lpSum(y[lcode] for lcode in lsoas_in_ward) == MAX_OFFICERS_PER_WARD

        # 5) Objective: maximize sum_{lcode}(risk_score[lcode] * y[lcode])
        prob += lpSum(raw_risk[lcode] * y[lcode] for lcode in lsoas_in_ward)

        # 6) Solve (CBC, silent)
        prob.solve(PULP_CBC_CMD(msg=False))

        # 7) Extract solution: if any varValue is None, treat as zero
        for lcode in lsoas_in_ward:
            val = y[lcode].varValue
            officers_count = int(val) if val is not None else 0
            hours = officers_count * MAX_PATROL_HOURS_PER_OFFICER

            allocation_results.append({
                "ward": ward,
                "lsoa": lcode,
                "risk_score": raw_risk[lcode],
                "officers_allocated": officers_count,
                "patrol_hours": hours,
            })

    return pd.DataFrame(allocation_results)

# Define column names for LSOA and ward
lsoa_col = "lsoa21cd"
ward_col = "lad22nm"
wards = find_london_wards(WARDS_PATH)
lsoas, lsoa_col = load_lsoa_data(LSOA_SHAPE_PATH)

# lsoa_predicted_risk_score = prediction_network()
lsoa_predicted_risk_score = pd.read_csv('prediction_results.csv')
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

allocation_df = allocate_resources(lsoas, ward_to_lsoas, lsoa_col, ward_col)
print(allocation_df.to_string())

# Merge officers_allocated from allocation_df into lsoas
allocation_df = allocation_df.rename(columns={"lsoa": lsoa_col})
map_df = lsoas.merge(
    allocation_df[[lsoa_col, "officers_allocated"]],
    on=lsoa_col,
    how="left"
)
map_df["officers_allocated"] = map_df["officers_allocated"].fillna(0).astype(int)

# Use lsoas_with_wards to build ward boundaries by dissolving LSOAs per ward
lsoas_with_wards = gpd.sjoin(lsoas, wards, how="left", predicate="within")
ward_boundaries = lsoas_with_wards.dissolve(by=ward_col).reset_index()[[ward_col, "geometry"]]

fig, ax = plt.subplots(1, 1, figsize=(12, 12))
# Plot LSOAs shaded by officers_allocated
map_df.plot(
    column="officers_allocated",
    cmap="OrRd",
    linewidth=0.1,
    edgecolor="grey",
    legend=True,
    legend_kwds={
        "label": "Number of Officers Allocated",
        "orientation": "horizontal",
        "shrink": 0.6
    },
    ax=ax
)
# Overlay ward boundaries derived from LSOAs
ward_boundaries.plot(
    ax=ax,
    facecolor="none",
    edgecolor="black",
    linewidth=1
)
ax.set_title("London LSOA Officers Allocated (Darker = More Officers) with Ward Boundaries", fontsize=16)
ax.axis("off")
plt.tight_layout()
plt.show()


