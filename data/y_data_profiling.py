# read data and create a y_data_profiling html report

from pathlib import Path

import pandas as pd
from icecream import ic
from ydata_profiling import ProfileReport

FILES = [
    Path("data/baumkataster_frankfurt_2021.parquet"),
    # Path("data/baumauswahl_v11_raw.parquet"),
    Path("data/baumkataster_berlin_2023.parquet"),
]


def create_report(parquet_path):
    ic("Working on", parquet_path)
    df = pd.read_parquet(parquet_path)
    profile = ProfileReport(df, title=f"Profiling Report {parquet_path.stem}")
    profile.to_file(f"data/{parquet_path.stem}_profiling_report.html")


if __name__ == "__main__":
    for file in FILES:
        try:
            create_report(file)
        except Exception as e:
            ic("Did not work", e)
