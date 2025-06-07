import geopandas as gpd
import pandas as pd
from pulp import *
from pulp import LpProblem, LpVariable, LpInteger, lpSum, LpMaximize, PULP_CBC_CMD
from time_series_prediction import prediction_network
import matplotlib.pyplot as plt
import os

# Constants
WARDS_PATH = 'data/London-wards-2018'
LSOA_SHAPE_PATH = 'data/LB_LSOA2021_shp'
MAX_OFFICERS_PER_WARD = 100
MAX_PATROL_HOURS_PER_OFFICER = 2
TOTAL_PATROL_HOURS_PER_WARD = MAX_OFFICERS_PER_WARD * MAX_PATROL_HOURS_PER_OFFICER  # 200
NEIGHBOR_WEIGHT_ALPHA = 0.05
mu = NEIGHBOR_WEIGHT_ALPHA


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


def compute_neighbor_pairs_with_weights(lsoas, lsoa_col, all_scores):
    """
    Compute adjacent LSOA pairs and assign risk-based weights.
    Weight w_ij = 1/(1+(r_i+r_j)/2) so borders between high-risk have lower penalty.
    """
    # spatial join for neighbors
    touches = gpd.sjoin(
        lsoas[[lsoa_col, 'geometry']],
        lsoas[[lsoa_col, 'geometry']],
        predicate='touches', how='inner'
    )[[f'{lsoa_col}_left', f'{lsoa_col}_right']]
    # clean
    touches = touches[touches[f'{lsoa_col}_left'] != touches[f'{lsoa_col}_right']]
    touches['pair'] = touches.apply(
        lambda r: tuple(sorted((r[f'{lsoa_col}_left'], r[f'{lsoa_col}_right']))), axis=1
    )
    touches = touches.drop_duplicates('pair')[[f'{lsoa_col}_left', f'{lsoa_col}_right']]
    pairs = list(touches.itertuples(index=False, name=None))
    # compute weights
    weights = {}
    for i, j in pairs:
        ri = all_scores.get(i, 0)
        rj = all_scores.get(j, 0)
        weights[(i, j)] = 1.0 / (1.0 + (ri + rj) / 2.0)
    return pairs, weights


def allocate_resources(lsoas: gpd.GeoDataFrame, ward_to_lsoas: dict, lsoa_col: str, ward_name_col: str) -> pd.DataFrame:
    records = []
    neighbor_pairs = compute_neighbor_pairs_with_weights(lsoas, lsoa_col,
                                                         lsoas.set_index(lsoa_col)["risk_score"].to_dict())[0]

    for ward, lsoa_list in ward_to_lsoas.items():
        subset = lsoas[lsoas[lsoa_col].isin(lsoa_list)]
        scores = dict(zip(subset[lsoa_col], subset["risk_score"]))

        N = len(lsoa_list)
        total_officers = MAX_OFFICERS_PER_WARD

        # even share and cap
        even_share = total_officers / N
        max_off = int(math.ceil(even_share * NEIGHBOR_WEIGHT_ALPHA))  # or any cap you prefer

        # 1. officer variables, forced ≥0 and ≤max_off
        officers = {
            l: LpVariable(f"off_{l}", lowBound=0, upBound=max_off, cat="Integer")
            for l in lsoa_list
        }

        # 2. pick only the neighbor‐pairs inside this ward
        edges = [(i, j) for i, j in neighbor_pairs if i in lsoa_list and j in lsoa_list]

        # 3. one slack var per edge
        slack = {
            (i, j): LpVariable(f"s_{i}_{j}", lowBound=0)
            for i, j in edges
        }

        # 4. build the problem
        prob = LpProblem(f"ward_{ward}", LpMaximize)

        # 4a. base risk objective (hours = officers * 2)
        base_obj = lpSum(
            scores[l] * officers[l] * MAX_PATROL_HOURS_PER_OFFICER
            for l in lsoa_list
        )
        # 4b. smoothing penalty
        smooth_penalty = lpSum(slack[(i, j)] for i, j in edges)

        prob += base_obj - mu * smooth_penalty

        # 5. total‐officer constraint
        prob += lpSum(officers.values()) == total_officers, "Total_Officers"

        # 6. absolute‐difference constraints
        for i, j in edges:
            prob += officers[i] - officers[j] <= slack[(i, j)]
            prob += officers[j] - officers[i] <= slack[(i, j)]

        # 7. solve
        prob.solve(PULP_CBC_CMD(msg=0))

        # 8. collect results
        for l in lsoa_list:
            num_off = int(officers[l].value() or 0)
            records.append({
                "ward": ward,
                "lsoa": l,
                "risk_score": scores[l],
                "officers_allocated": num_off,
                "patrol_hours": num_off * MAX_PATROL_HOURS_PER_OFFICER
            })

    return pd.DataFrame(records)


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
# Calculate total officers and patrol hours per ward
ward_summary = allocation_df.groupby("ward").agg(
    total_officers_allocated=("officers_allocated", "sum"),
    total_patrol_hours=("patrol_hours", "sum")
).reset_index()

print(ward_summary.to_string())
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
