import logging
from typing import Optional


def merge_data_sources(data_sources: list, config: dict):
    import pandas as pd
    time_column = config["time_column"]
    timeframe = config["timeframe"]

    for ds in data_sources:
        df = ds.get("df")
        if time_column in df.columns:
            df = df.set_index(time_column)
        elif getattr(df.index, 'name', None) == time_column:
            pass
        else:
            print(f"ERROR: Timestamp column is absent.")
            return None

        # Add prefix if not already there
        prefix = ds.get('column_prefix', "")
        if prefix:
            df.columns = [
                f"{prefix}_{col}" if not col.startswith(f"{prefix}_") else col
                for col in df.columns
            ]

        ds["start"] = df.index[0]
        ds["end"] = df.index[-1]
        ds["df"] = df

    # Create common (main) index and empty data frame
    range_start = min([ds["start"] for ds in data_sources])
    range_end = min([ds["end"] for ds in data_sources])
    index = pd.date_range(range_start, range_end, freq=timeframe)
    df_out = pd.DataFrame(index=index)
    df_out.index.name = time_column

    for ds in data_sources:
        df_out = df_out.join(ds["df"], how="left")

    # Interpolate numeric columns if enabled
    merge_interpolate = config.get("merge_interpolate", False)
    if merge_interpolate:
        num_columns = df_out.select_dtypes(include=[float, int]).columns.tolist()
        for col in num_columns:
            df_out[col] = df_out[col].interpolate()

    return df_out

