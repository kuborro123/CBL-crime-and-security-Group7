import geopandas as gpd
import numpy as np
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
NEIGHBOR_WEIGHT_ALPHA = 0.50
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


def plot_allocation_map(lsoas, allocation_df, wards, ward_col, lsoa_col):
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
    neighbor_pairs, _ = compute_neighbor_pairs_with_weights(
        lsoas, lsoa_col, lsoas.set_index(lsoa_col)["risk_score"].to_dict()
    )

    for ward, lsoa_list in ward_to_lsoas.items():
        ward_lsoas = lsoas[lsoas[lsoa_col].isin(lsoa_list)].copy()
        ward_lsoas["log_risk"] = ward_lsoas["risk_score"].apply(np.log1p)
        scores = dict(zip(ward_lsoas[lsoa_col], ward_lsoas["log_risk"]))
        ward_lsoas["risk_score"] = ward_lsoas["risk_score"].clip(lower=0)

        # 2. Apply the logarithmic transformation on the cleaned scores
        ward_lsoas["log_risk"] = ward_lsoas["risk_score"].apply(np.log1p)
        scores = dict(zip(ward_lsoas[lsoa_col], ward_lsoas["log_risk"]))

        # 1. Calculate risk score average and standard deviation for the ward
        risk_scores = ward_lsoas["risk_score"]
        ward_avg = risk_scores.mean()
        ward_std = risk_scores.std()

        # Handle cases where std is 0 to avoid division by zero or pointless checks
        if pd.isna(ward_std) or ward_std == 0:
            threshold = ward_avg
        else:
            threshold = ward_avg + (2.5 * ward_std)

        # 2. Identify outlier LSOAs (score > avg + 3*std)
        outlier_lsoas = ward_lsoas[ward_lsoas["risk_score"] > threshold]

        lsoas_for_optimization = lsoa_list.copy()
        remaining_officers = MAX_OFFICERS_PER_WARD

        # 3. Pre-allocate 8 officers to each outlier and remove them from the pool
        # We sort outliers by risk to prioritize the highest-risk ones if officer budget is tight
        for lsoa_code in outlier_lsoas.sort_values("risk_score", ascending=False)[lsoa_col]:
            if remaining_officers >= 8:
                num_off = 8
                records.append({
                    "ward": ward,
                    "lsoa": lsoa_code,
                    "risk_score": scores[lsoa_code],
                    "officers_allocated": num_off,
                    "patrol_hours": num_off * MAX_PATROL_HOURS_PER_OFFICER
                })
                remaining_officers -= num_off
                lsoas_for_optimization.remove(lsoa_code)

        # 4. Run optimization only if there are remaining officers and LSOAs
        if not lsoas_for_optimization or remaining_officers <= 0:
            # If all LSOAs were outliers or no officers are left, skip optimization for this ward
            continue

        subset_scores = {l: scores[l] for l in lsoas_for_optimization}
        N = len(lsoas_for_optimization)
        total_officers = remaining_officers  # Use remaining officers

        even_share = total_officers / N if N > 0 else 0
        # Cap allocation to avoid one LSOA taking all resources
        max_off = int(math.ceil(even_share * NEIGHBOR_WEIGHT_ALPHA * 2)) if even_share > 0 else total_officers

        officers = {
            l: LpVariable(f"off_{l}", lowBound=0, upBound=max_off, cat="Integer")
            for l in lsoas_for_optimization
        }

        edges = [(i, j) for i, j in neighbor_pairs if i in lsoas_for_optimization and j in lsoas_for_optimization]
        slack = {
            (i, j): LpVariable(f"s_{i}_{j}", lowBound=0)
            for i, j in edges
        }

        prob = LpProblem(f"ward_{ward}_opt", LpMaximize)
        base_obj = lpSum(
            subset_scores[l] * officers[l] * MAX_PATROL_HOURS_PER_OFFICER
            for l in lsoas_for_optimization
        )
        smooth_penalty = lpSum(slack[(i, j)] for i, j in edges)
        prob += base_obj - mu * smooth_penalty
        prob += lpSum(officers.values()) == total_officers, "Total_Officers"

        for i, j in edges:
            prob += officers[i] - officers[j] <= slack[(i, j)]
            prob += officers[j] - officers[i] <= slack[(i, j)]

        prob.solve(PULP_CBC_CMD(msg=0))

        for l in lsoas_for_optimization:
            num_off = int(officers[l].value() or 0)
            records.append({
                "ward": ward,
                "lsoa": l,
                "risk_score": subset_scores[l],
                "officers_allocated": num_off,
                "patrol_hours": num_off * MAX_PATROL_HOURS_PER_OFFICER
            })

    return pd.DataFrame(records)


def create_patrol_schedule(allocation_df: pd.DataFrame, neighbor_pairs: list) -> pd.DataFrame:
    """
    Creates a patrol schedule based on officer allocation, prioritizing neighbor
    coordination and then ensuring homogeneous coverage for all others.

    Args:
        allocation_df: DataFrame from the allocate_resources function.
        neighbor_pairs: A list of tuples, where each tuple is a pair of adjacent LSOAs.

    Returns:
        A DataFrame with the schedule, including columns for each time block.
    """
    # 1. Setup and Initialization
    time_blocks = [
        "06:00-08:00", "08:00-10:00", "10:00-12:00", "12:00-14:00",
        "14:00-16:00", "16:00-18:00", "18:00-20:00", "20:00-22:00"
    ]

    # Prepare the output DataFrame
    schedule_df = allocation_df.copy()
    for block in time_blocks:
        schedule_df[block] = 0

    # Use LSOA code as the index for easy row lookup and modification
    schedule_df = schedule_df.set_index("lsoa")

    lsoas_to_schedule = set(schedule_df.index)

    # 2. Coordinated Neighbor Scheduling (Low-Allocation Pairs)
    # Convert neighbor pairs to a set for efficient lookup
    neighbor_set = {tuple(sorted(p)) for p in neighbor_pairs}

    for lsoa1, lsoa2 in neighbor_set:
        # Check if both LSOAs are in our list and haven't been scheduled yet
        if lsoa1 in lsoas_to_schedule and lsoa2 in lsoas_to_schedule:
            officers1 = schedule_df.loc[lsoa1, "officers_allocated"]
            officers2 = schedule_df.loc[lsoa2, "officers_allocated"]

            # Define the "low allocation" threshold for coordination
            if 0 < officers1 <= 2 and 0 < officers2 <= 2:
                # Pool the officers and create a non-overlapping schedule

                # Schedule LSOA 1 in the earliest available slots
                i = 0
                for _ in range(officers1):
                    schedule_df.loc[lsoa1, time_blocks[i]] += 1
                    i += 1

                # Schedule LSOA 2 in the next available slots
                for _ in range(officers2):
                    # Ensure we don't go out of bounds
                    if i < len(time_blocks):
                        schedule_df.loc[lsoa2, time_blocks[i]] += 1
                        i += 1

                # Mark both as "done"
                lsoas_to_schedule.remove(lsoa1)
                lsoas_to_schedule.remove(lsoa2)

    # 3. Independent Homogeneous Scheduling (for all remaining LSOAs)
    for lsoa in lsoas_to_schedule:
        officers_to_assign = schedule_df.loc[lsoa, "officers_allocated"]

        if officers_to_assign > 0:
            time_block_index = 0
            while officers_to_assign > 0:
                # Get the current time block and assign an officer
                block = time_blocks[time_block_index]
                schedule_df.loc[lsoa, block] += 1

                # Decrement officers and move to the next time block
                officers_to_assign -= 1
                time_block_index = (time_block_index + 1) % len(time_blocks)

    return schedule_df.reset_index()


def plot_schedule_maps(schedule_df: pd.DataFrame, lsoas: gpd.GeoDataFrame, wards: gpd.GeoDataFrame, ward_col: str,
                       lsoa_col: str):
    """
    Generates and displays a separate map for each time block in the schedule.

    Args:
        schedule_df: The final DataFrame containing the schedule.
        lsoas: The base GeoDataFrame for LSOA geometries.
        wards: The base GeoDataFrame for ward geometries.
        ward_col: The name of the ward column.
        lsoa_col: The name of the LSOA column.
    """
    # Identify the time block columns to be plotted
    time_blocks = [
        "06:00-08:00", "08:00-10:00", "10:00-12:00", "12:00-14:00",
        "14:00-16:00", "16:00-18:00", "18:00-20:00", "20:00-22:00"
    ]

    # Merge the geometries into the schedule dataframe once
    map_df = lsoas.merge(schedule_df, on=lsoa_col, how="left")

    # Pre-calculate the ward boundaries
    lsoas_with_wards = gpd.sjoin(lsoas, wards, how="left", predicate="within")

    # Use the correct ward column after the join
    if f"{ward_col}_right" in lsoas_with_wards.columns:
        ward_boundaries = lsoas_with_wards.dissolve(by=f"{ward_col}_right").reset_index()[
            [f"{ward_col}_right", "geometry"]]
    else:
        ward_boundaries = lsoas_with_wards.dissolve(by=ward_col).reset_index()[[ward_col, "geometry"]]

    # Find the maximum allocation in any time slot to use for a consistent legend
    max_officers_in_slot = map_df[time_blocks].max().max() if not map_df[time_blocks].empty else 1.0

    print("\nGenerating schedule maps...")
    # Loop through each time block and create a separate plot
    for block in time_blocks:
        # Fill any NaNs in the current block's column with 0
        map_df[block] = map_df[block].fillna(0).astype(int)

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        map_df.plot(
            column=block,
            cmap="OrRd",
            linewidth=0.1,
            edgecolor="grey",
            legend=True,
            legend_kwds={
                "label": f"Number of Officers Allocated ({block})",
                "orientation": "horizontal",
                "shrink": 0.6
            },
            ax=ax,
            # Use vmin/vmax for a consistent color scale across all maps
            vmin=0,
            vmax=max_officers_in_slot
        )

        ward_boundaries.plot(
            ax=ax,
            facecolor="none",
            edgecolor="black",
            linewidth=1
        )

        ax.set_title(f"London Officer Patrol Schedule: {block}", fontsize=16)
        ax.axis("off")
        plt.tight_layout()
        plt.show()


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
# Use the correct ward column name after the join
if f"{ward_col}_right" in lsoas_with_wards.columns:
    ward_col_grouped = f"{ward_col}_right"
else:
    ward_col_grouped = ward_col
ward_to_lsoas = lsoas_with_wards.groupby(ward_col_grouped)[lsoa_col].apply(list).to_dict()

allocation_df = allocate_resources(lsoas, ward_to_lsoas, lsoa_col, ward_col_grouped)
schedule = create_patrol_schedule(allocation_df, compute_neighbor_pairs_with_weights(lsoas, lsoa_col, lsoas.set_index(lsoa_col)["risk_score"].to_dict())[0])
print(schedule.to_string())

# --- FIX: Rename the 'lsoa' column to the correct name before plotting ---
schedule = schedule.rename(columns={"lsoa": lsoa_col})
schedule.to_csv('schedule_output.csv', index=False)
plot_schedule_maps(schedule, lsoas, wards, ward_col_grouped, lsoa_col)
