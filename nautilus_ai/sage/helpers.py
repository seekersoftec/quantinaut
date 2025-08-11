import logging
from typing import Optional


def merge_data_sources(data_sources: list, config: dict):
    import pandas as pd
    time_column = config["time_column"]
    freq = config["freq"]

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
    index = pd.date_range(range_start, range_end, freq=freq)
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



def connect_mt5(mt5_account_id: Optional[int] = None, mt5_password: Optional[str] = None, mt5_server: Optional[str] = None, **kwargs):
    """
    Initializes the MetaTrader 5 connection and attempts to log in with the provided credentials.
    """
    import MetaTrader5 as mt5
    
    log = logging.getLogger('mt5')

    
    # Initialize MetaTrader 5 connection
    if not mt5.initialize():
        log.error(f"initialize() failed, error code = {mt5.last_error()}")
        return False
    log.info(f"MT5 Initialized. Version: {mt5.version()}")

    if mt5_account_id and mt5_password and mt5_server:
        authorized = mt5.login(int(mt5_account_id), password=str(mt5_password), server=str(mt5_server), **kwargs)
        if not authorized:
            log.error(f"MT5 Login failed for account #{mt5_account_id}, error code: {mt5.last_error()}")
            mt5.shutdown()
            return False
    return True

