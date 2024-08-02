import re

import numpy as np
import pandas as pd


def read_trees_ber() -> pd.DataFrame:
    """Read Berlin tree data from local parquet file.

    Reads data, transforms categorical columns.

    Returns:
        pd.DataFrame: Dataframe with Berlin tree data.
    """
    raw_data = "../data/baumkataster_berlin_2023.parquet"
    data = pd.read_parquet(raw_data)

    categoricals = [
        "bezirk",
        "art_dtsch",
        "art_bot",
        "gattung_deutsch",
        "gattung",
        "standortnr",
        "kennzeich",
        "namenr",
        "eigentuemer",
    ]
    data[categoricals] = data[categoricals].astype("category")

    return data


def prepare_trees_ber(thres_rare=20) -> pd.DataFrame:
    """Prepares Berlin tree data for usage.

    Reads Berlin tree data with `prepare_trees_ber()`, removes outliers,
      splits the data into labeled and unlabeled along 'Pflanzjahr',
      replaces rare categorical values, and replaces NAs for
      'gattung_deutsch'.

    Args:
        thres_rare (int, optional): Threshold of rare values in
          categorical columns. Categories with fewer occurrences are
          replaced with the category 'rare'. Defaults to 100.

    Returns:
        pd.DataFrame: Dataframe with prepared Berlin tree data.
    """
    data = read_trees_ber()

    del data["baumid"]
    del data["standortnr"]
    del data["standalter"]

    data["namenr_nonum"] = data["namenr"].apply(
        lambda key: re.sub(r"\d+(-\d+)?|\b[IVXLCDM]+\b", "", key).strip()
    )
    data["namenr_nonum"] = data["namenr_nonum"].astype("category")
    data["lon_section"] = data["lon"].round(1).astype("category")
    data["lat_section"] = data["lat"].round(1).astype("category")
    data["lat_lon_tile"] = data["lat_section"].astype(str) + "-" + data["lon_section"].astype(str)
    data["lat_lon_tile"] = data["lat_lon_tile"].astype("category")

    max_stammumfg = 1665  # 16,65m / π = 5,3m
    data.loc[data["stammumfg"] > max_stammumfg, "stammumfg"] = np.nan

    max_kronedurch = 70
    data.loc[data["kronedurch"] >= max_kronedurch, "kronedurch"] = np.nan

    max_baumhoehe = 100
    data.loc[data["baumhoehe"] >= max_baumhoehe, "baumhoehe"] = np.nan

    min_pflanzjahr = 1400
    max_pflanzjahr = 2023
    data.loc[
        (data["pflanzjahr"] <= min_pflanzjahr) | (data["pflanzjahr"] >= max_pflanzjahr),
        "pflanzjahr",
    ] = np.nan

    data["hoehe_zu_krone"] = data["baumhoehe"] / data["kronedurch"].replace(0, np.nan)
    data["hoehe_zu_stamm"] = data["baumhoehe"] / data["stammumfg"].replace(0, np.nan)

    data["gattung"] = data["gattung"].cat.rename_categories({"Abies": "ABIES"})

    genus_cat_mapping = pd.read_parquet("../data/berlin_genus_category_mapping.parquet")
    data = data.merge(genus_cat_mapping, on="gattung", how="left")
    data["gattung"] = data["gattung"].astype("category")
    data["baumart"] = data["baumart"].astype("category")

    data["gattung_deutsch"] = data["gattung_deutsch"].cat.add_categories(["unknown"])
    data["gattung_deutsch"] = data["gattung_deutsch"].fillna("unknown")

    replacement_cols = ["art_dtsch", "art_bot", "gattung_deutsch", "gattung"]
    threshold = 1000
    for col in replacement_cols:
        data[col + "_infrequent"] = data[col].map(data[col].value_counts() < threshold)
        data[col + "_infrequent"] = data[col + "_infrequent"].astype("bool")

    unlabeled = data[data["pflanzjahr"].isna()].copy()
    labeled = data[data["pflanzjahr"].notna()].copy()

    cat_cols = labeled.select_dtypes(include="category").columns.tolist()
    for cat_col in cat_cols:
        counts = labeled[cat_col].value_counts()
        rare = counts[counts < thres_rare].index.to_list()

        if len(rare) > 0:
            labeled[cat_col] = labeled[cat_col].astype("object")
            labeled.loc[labeled.query(f"{cat_col} == @rare").index, cat_col] = "rare"
            labeled[cat_col] = labeled[cat_col].astype("category")
            labeled[cat_col] = labeled[cat_col].cat.remove_unused_categories()

    return data, labeled, unlabeled


def create_ber_subset() -> None:
    """Creates a subset with three species of trees from the Berlin tree dataset."""
    _, labeled, _ = prepare_trees_ber()
    subset = labeled[labeled["gattung_deutsch"].isin(["EIBE", "APFEL", "HASEL"])]

    for col in subset.columns:
        if isinstance(subset[col], pd.CategoricalDtype):
            subset[col] = subset[col].cat.remove_unused_categories()

    subset.to_parquet("../data/subset_baumkataster_berlin_2023.parquet")
