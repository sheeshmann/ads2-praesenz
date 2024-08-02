"""Module to read datasets repetitively."""

import pandas as pd


def read_trees_fra() -> pd.DataFrame:
    """Reads the Frankfurt Tree dataset and applies minor preprocessing.

    As the data originates from 2021, trees older than that are removed.
    The column names are casted to lowercase. The categorial datatype is
    applied to categories.

    Returns:
        pd.DataFrame: Dataframe holding the Frankfurt tree dataset.
    """
    raw_data = "../data/baumkataster_frankfurt_2021.parquet"
    data = pd.read_parquet(raw_data)
    data.columns = [x.lower() for x in data.columns.to_list()]
    data = data[data["pflanzjahr"] < 2022]

    categoricals = [
        "gattungart",
        "gattung",
        "ga_lang",
        "gebiet",
        "strasse",
        "standort",
        "baum_statu",
    ]
    data[categoricals] = data[categoricals].astype("category")

    data = data[data["st_durchm"] < 1000]

    return data
