import re
import os
import time
import logging
import pytz
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import MetaTrader5 as mt5

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Configuration ---
DEFAULT_BAR_CHUNK_SIZE = 10000  # How many bars worth of duration to request in each chunk
DEFAULT_TICK_CHUNK_SIZE = 5 # How many ticks worth of duration to request in each chunk
RATE_LIMIT_DELAY = 0.1 # Small delay between requests (seconds)

# ---------------------

print("MetaTrader5 package author: ", mt5.__author__)
print("MetaTrader5 package version: ", mt5.__version__)


def mt5_freq_from_pandas(freq: str) -> int:
    """
    Dynamically map pandas frequency strings to MetaTrader5 API timeframe constants.

    Handles inputs like '1min', '15min', '1h', '4h', '1D', 'D', '1W', 'W', '1MS', 'MS'.

    :param freq: pandas frequency string (e.g., '5min', '1h', '1D').
                 See https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
    :return: Corresponding MetaTrader5 TIMEFRAME_* constant (integer).
             See https://www.mql5.com/en/docs/integration/python_metatrader5/
    :raises ValueError: If the frequency string is not recognized or the corresponding
                        MT5 constant cannot be found.
    """
    # Map Pandas units (lowercase) to MT5 prefixes and whether they always imply '1'
    unit_map = {
        'min': ('M', False),
        'h':   ('H', False),
        'd':   ('D', True),
        'w':   ('W', True),
        'ms':  ('MN', True), # Month Start maps to MN1
    }

    # Try to match pattern: optional number + unit letters
    match = re.fullmatch(r"(\d+)?([A-Za-z]+)", str(freq))

    if not match:
        raise ValueError(f"Input frequency '{freq}' does not match expected format (e.g., '1min', '4h', '1D').")

    num_str, unit_pandas_raw = match.groups()
    unit_pandas = unit_pandas_raw.lower() # Normalize unit to lower case for map lookup

    # Find the corresponding MT5 unit info
    mt5_prefix, is_always_one = None, False
    found_unit = False
    if unit_pandas == 'min':
         mt5_prefix, is_always_one = unit_map['min']
         found_unit = True
    elif unit_pandas == 'h':
         mt5_prefix, is_always_one = unit_map['h']
         found_unit = True
    # Use original case for D, W, MS check as they are distinct in Pandas
    elif unit_pandas_raw == 'D':
         mt5_prefix, is_always_one = unit_map['d'] # map key is lowercase
         found_unit = True
    elif unit_pandas_raw == 'W':
         mt5_prefix, is_always_one = unit_map['w'] # map key is lowercase
         found_unit = True
    elif unit_pandas_raw == 'MS':
         mt5_prefix, is_always_one = unit_map['ms'] # map key is lowercase
         found_unit = True

    if not found_unit:
         raise ValueError(f"Unsupported Pandas frequency unit '{unit_pandas_raw}' in '{freq}'.")

    # Determine the number part
    if is_always_one:
        number = 1
    elif num_str:
        number = int(num_str)
    else:
        # If number is missing for min/h (e.g., 'h'), assume 1
        number = 1

    # Construct the MT5 constant name (e.g., "TIMEFRAME_M15", "TIMEFRAME_H4", "TIMEFRAME_D1")
    mt5_constant_name = f"TIMEFRAME_{mt5_prefix}{number}"

    # Retrieve the constant value from the mt5 module
    try:
        return getattr(mt5, mt5_constant_name)
    except AttributeError:
        # Provide a more informative error if the constant doesn't exist
        supported_timeframes = [tf for tf_name, tf in mt5.__dict__.items() if tf_name.startswith('TIMEFRAME_')]
        raise ValueError(
            f"Could not find or map MetaTrader5 constant '{mt5_constant_name}' for frequency '{freq}'. "
            f"Check if this timeframe is supported by the MetaTrader5 library/API. "
            f"Available TIMEFRAME constants might include: {sorted(list(set(supported_timeframes)))}"
        )


def get_timedelta_for_mt5_timeframe(mt5_timeframe: int, count: int) -> timedelta:
    """
    Calculate the total duration corresponding to 'count' bars
    of the specified MT5 timeframe constant.

    Internally maintains a cache of parsed timeframe details
    and a compiled regex for parsing attribute names.

    :param mt5_timeframe: MT5 constant (e.g., mt5.TIMEFRAME_M15)
    :param count: Number of bars
    :return: timedelta representing the aggregated duration
    :raises ValueError: If the timeframe is unknown or unsupported
    """
    # Initialize static attributes on the function for cache and pattern
    if not hasattr(get_timedelta_for_mt5_timeframe, "_pattern"):
        # Compile regex once
        get_timedelta_for_mt5_timeframe._pattern = re.compile(r"TIMEFRAME_([A-Z]+)(\d+)$")
        # Build cache mapping MT5 timeframe constants to (name, unit, number)
        cache: dict[int, tuple[str, str, int]] = {}
        for attr_name, attr_value in mt5.__dict__.items():
            if not attr_name.startswith("TIMEFRAME_") or not isinstance(attr_value, int):
                continue
            match = get_timedelta_for_mt5_timeframe._pattern.match(attr_name)
            if match:
                unit_prefix, number_str = match.groups()
                cache[attr_value] = (attr_name, unit_prefix, int(number_str))
            elif attr_name == "TIMEFRAME_MN1":
                # Special case for monthly timeframe without explicit number
                cache[attr_value] = (attr_name, "MN", 1)
        get_timedelta_for_mt5_timeframe._cache = cache
        logger.debug("Initialized MT5 timeframe pattern and cache")

    # Retrieve static attributes
    pattern = get_timedelta_for_mt5_timeframe._pattern
    cache = get_timedelta_for_mt5_timeframe._cache

    details = cache.get(mt5_timeframe)
    if details is None:
        raise ValueError(f"Unknown MetaTrader5 timeframe constant: {mt5_timeframe}")

    name, unit_prefix, number = details

    # Mapping of unit prefix to a factory function returning a timedelta
    unit_to_timedelta = {
        'M': lambda n, c: timedelta(minutes=n * c),
        'H': lambda n, c: timedelta(hours=n * c),
        'D': lambda n, c: timedelta(days=n * c),
        'W': lambda n, c: timedelta(weeks=n * c),
        'MN': lambda n, c: timedelta(days=n * c * 30.5),  # approximate month
    }

    factory = unit_to_timedelta.get(unit_prefix)
    if factory is None:
        raise ValueError(f"Unsupported timeframe unit '{unit_prefix}' derived from {name}")

    if unit_prefix == 'MN':
        logger.warning("Using approximate duration of 30.5 days for monthly timeframes.")

    return factory(number, count)


def parse_or_default(date_str, fmt, tz, default_dt):
    if date_str:
        dt = datetime.strptime(date_str, fmt)
        return dt.replace(tzinfo=tz)
    return default_dt

def connect_mt5(mt5_account_id: Optional[int] = None, mt5_password: Optional[str] = None, mt5_server: Optional[str] = None, **kwargs):
    """
    Initializes the MetaTrader 5 connection and attempts to log in with the provided credentials.
    """
    # Initialize MetaTrader 5 connection
    if not mt5.initialize():
        logger.error(f"initialize() failed, error code = {mt5.last_error()}")
        return False
    logger.info(f"MT5 Initialized. Version: {mt5.version()}")

    if mt5_account_id and mt5_password and mt5_server:
        authorized = mt5.login(int(mt5_account_id), password=str(mt5_password), server=str(mt5_server), **kwargs)
        if not authorized:
            logger.error(f"MT5 Login failed for account #{mt5_account_id}, error code: {mt5.last_error()}")
            mt5.shutdown()
            return False
    return True

# -------------------------------------------------

def scrape_and_save_candles(data_sources: List[Dict[str, Any]], mt5_account_id: Optional[int] = None, mt5_password: Optional[str] = None, mt5_server: Optional[str] = None, timeframe: str = "1h", time_column: str = "timestamp", since: Union[int, str] = None, until: Union[int, str] = None, **kwargs):
    """
    Retrieving historic klines from MetaTrader5 server incrementally using copy_rates_range
    with calculated chunk durations. Downloads data from the last record in the existing file
    (or a historical start date) up to the current time, fetching in duration-based chunks.
    """
    data_path = Path(kwargs.get("data_folder", "data"))
    download_max_rows = kwargs.get("download_max_rows", 0)

    script_start_time = datetime.now()

    print(f"Pandas frequency: {timeframe}")

    mt5_timeframe = mt5_freq_from_pandas(timeframe)
    # Use timeframe_description for clearer output if available
    try:
        tf_description = mt5.timeframe_description(mt5_timeframe)
        print(f"MetaTrader5 frequency: {tf_description} ({mt5_timeframe})")
    except AttributeError: # Handle older MT5 versions potentially lacking this func
         print(f"MetaTrader5 frequency: {mt5_timeframe}")


    # Define the timezone for MT5 (usually UTC)
    timezone = pytz.timezone("Etc/UTC")
    # Define a default historical start if no file exists => 2014 | 2017 | 2024
    historical_start_date = parse_or_default(
        since,
        "%Y-%m-%dT%H:%M:%SZ",
        timezone,
        datetime(2017, 1, 1, tzinfo=timezone),
    )
    end_date = parse_or_default(
        until,
        "%Y-%m-%dT%H:%M:%SZ",
        timezone,
        datetime.now(timezone),
    )
    
    # Connect to trading account 
    if mt5_account_id and mt5_password and mt5_server:
        authorized = connect_mt5(mt5_account_id, password=str(mt5_password), server=str(mt5_server))
        if authorized:
            print("MT5 Login successful.")
            account_info = mt5.account_info()
            if account_info:
                print(f"Account Info: Login={account_info.login}, Server={account_info.server}, Balance={account_info.balance}")
            else:
                 print(f"Could not retrieve account info. Error: {mt5.last_error()}")
        else:
            print(f"MT5 Login failed for account #{mt5_account_id}, error code: {mt5.last_error()}")
            mt5.shutdown()
            return
    else:
        print("MT5 credentials not fully provided in config. Proceeding without login (might affect available symbols/data).")

    print(f"Terminal Info: {mt5.terminal_info()}")


    # --- Loop through data sources ---

    processed_symbols = []

    for ds in data_sources:
        quote = str(ds.get("symbol"))
        data_type = str(ds.get("type")).lower()

        if not quote:
            print(f"ERROR: {quote} is not specified in data_sources.")
            continue


        print(f"\n--- Processing symbol: {quote} ---")

        file_path = data_path / "mt5"
        file_path.mkdir(parents=True, exist_ok=True)
        file_name = (file_path / f"{quote}_{timeframe}_klines".lower()).with_suffix(".csv")
        chunk_size = int(ds.get("chunk_size", DEFAULT_BAR_CHUNK_SIZE))


        if data_type == "ticks":
            file_name = (file_path / f"{quote}_ticks".lower()).with_suffix(".csv")
            chunk_size = int(ds.get("chunk_size", DEFAULT_TICK_CHUNK_SIZE))


        existing_df = pd.DataFrame()
        start_dt = historical_start_date


        # Check if file exists and load data
        if file_name.is_file():
            try:
                print(f"Loading existing data from: {file_name}")
                # Specify date format for potentially faster parsing if consistent
                existing_df = pd.read_csv(file_name, parse_dates=[time_column], date_format='ISO8601')
                # Ensure timezone is set correctly after parsing
                if pd.api.types.is_datetime64_any_dtype(existing_df[time_column]) and existing_df[time_column].dt.tz is None:
                     existing_df[time_column] = existing_df[time_column].dt.tz_localize('UTC')
                elif pd.api.types.is_datetime64_any_dtype(existing_df[time_column]):
                     existing_df[time_column] = existing_df[time_column].dt.tz_convert('UTC')
                else: # Fallback if parsing failed or column is not datetime
                    print(f"Warning: Column '{time_column}' not parsed as datetime. Attempting conversion.")
                    existing_df[time_column] = pd.to_datetime(existing_df[time_column], errors='coerce', utc=True)

                existing_df = existing_df.dropna(subset=[time_column]) # Drop rows where conversion failed

                if not existing_df.empty:
                    # Sort just in case file wasn't sorted
                    existing_df = existing_df.sort_values(by=time_column)
                    # Start downloading from the timestamp of the last record
                    start_dt = existing_df[time_column].iloc[-1]
                    print(f"Existing file found. Will download data starting from {start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                else:
                     print("Existing file was empty or had invalid dates after loading. Starting from historical date.")
                     existing_df = pd.DataFrame() # Reset to empty
            except Exception as e:
                print(f"Error loading existing file {file_name}: {e}. Starting from historical date.")
                existing_df = pd.DataFrame() # Reset to empty
        else:
            print(f"File not found. Starting download from {historical_start_date.strftime('%Y-%m-%d %H:%M:%S %Z')}.")
            start_dt = historical_start_date # Ensure start_dt is set


        # Define end point for download (now)
        end_dt = end_date

        # Check if symbol is available
        symbol_info = mt5.symbol_info(quote)
        if not symbol_info:
            print(f"Symbol {quote} not found or not available in MT5 terminal. Skipping. Error: {mt5.last_error()}")
            continue
        if data_type == "ticks" and not symbol_info.trade_tick_size:
            print(f"Ticks data is not available for {quote}. Skipping. Error: {mt5.last_error()}")
            os.remove(file_name)
            continue
        print(f"Symbol {quote} found in MT5.")


        all_klines_list = []
        current_start_dt = start_dt

        print(f"Starting download loop for {quote} from {current_start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} to {end_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}...")

        # --- Download Loop using copy_rates_range or copy_ticks_range with calculated duration ---
        while current_start_dt < end_dt:
            try:
                # Calculate the duration for chunk_size bars or ticks


                chunk_duration = get_timedelta_for_mt5_timeframe(mt5_timeframe, chunk_size)
            except ValueError as e:
                 print(f"Error calculating duration: {e}. Stopping download for {quote}.")
                 break

            # Calculate the temporary end date for this chunk request
            temp_end_dt = current_start_dt + chunk_duration

            # Ensure the temporary end date doesn't go beyond the overall end date
            temp_end_dt = min(temp_end_dt, end_dt)

            # Add a small buffer (e.g., 1 second) to start_dt for the request
            # to definitively exclude the current_start_dt bar itself in the range request.
            request_start_dt = current_start_dt + timedelta(seconds=1)

            # Avoid making a request if the adjusted start is already >= temp_end
            if request_start_dt >= temp_end_dt:
                 print(f"  Skipping request: Calculated range is empty or invalid ({request_start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} to {temp_end_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}).")
                 break # Likely means we are caught up

            print(f"  Fetching range from {request_start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} to {temp_end_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}...")

            # Use copy_rates_range or copy_ticks_range
            if data_type == "ticks":
                rates = mt5.copy_ticks_range(quote, request_start_dt, temp_end_dt, mt5.COPY_TICKS_ALL)
                if rates is None:
                    print(f"  mt5.copy_ticks_range returned None. Error: {mt5.last_error()}. Stopping download for {quote}.")
                    break
            else:
                rates = mt5.copy_rates_range(quote, mt5_timeframe, request_start_dt, temp_end_dt)
                if rates is None:
                    print(f"  mt5.copy_rates_range returned None. Error: {mt5.last_error()}. Stopping download for {quote}.")
                    break

            if len(rates) == 0:
                print("  No data returned in this range. Download may be complete or data gap.")
                # If no data, advance start time past this chunk's end to avoid getting stuck
                current_start_dt = temp_end_dt
                if current_start_dt >= end_dt:
                    print("  Reached end date after empty range.")
                    break
                else:
                    print(f"  Advancing start time to {current_start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} and continuing.")
                    time.sleep(RATE_LIMIT_DELAY) # Still pause slightly
                    continue # Try the next chunk

            chunk_df = pd.DataFrame(rates)
            if data_type == "ticks":
                # Convert 'time_msc' (Unix milliseconds) to datetime objects (UTC)
                chunk_df[time_column] = pd.to_datetime(chunk_df['time_msc'], unit='ms', utc=True)
            else:
                # Convert 'time' (Unix seconds) to datetime objects (UTC)
                chunk_df[time_column] = pd.to_datetime(chunk_df['time'], unit='s', utc=True)

            # --- IMPORTANT: Filtering is no longer needed here ---
            # Since we requested data *starting after* current_start_dt using request_start_dt,
            # the check `chunk_df = chunk_df[chunk_df[time_column] > current_start_dt]`
            # is redundant and can be removed.

            # if chunk_df.empty: # This check is effectively handled by len(rates) == 0 now
            #      print("  No new bars found in the fetched chunk (after filtering). Stopping.")
            #      break

            all_klines_list.append(chunk_df)
            last_bar_time_in_chunk = chunk_df[time_column].iloc[-1]
            print(f"  Fetched {len(chunk_df)} bars. Last timestamp in chunk: {last_bar_time_in_chunk.strftime('%Y-%m-%d %H:%M:%S %Z')}")

            # Update the start time for the next chunk request to be the end time of THIS chunk's last bar
            current_start_dt = last_bar_time_in_chunk

            # Check if we've downloaded past the intended end time (redundant check, loop condition handles it)
            # if current_start_dt >= end_dt:
            #     print("  Reached or passed target end time. Download complete.")
            #     break

            # Small delay before next request
            time.sleep(RATE_LIMIT_DELAY)

        # --- Combine and Process Data ---
        if not all_klines_list:
            print(f"No new data downloaded for {quote}.")
            if existing_df.empty:
                print(f"No existing or new data for {quote}. Skipping save.")
                continue
            else:
                print("Saving existing data only (no updates).")
                final_df = existing_df # Use existing data if no new data was fetched
        else:
            print("Combining downloaded data...")
            new_df = pd.concat(all_klines_list, ignore_index=True)

            # Combine existing and new data
            if not existing_df.empty:
                final_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                final_df = new_df

            print("Processing combined data (duplicates, sorting, columns)...")
            if data_type == "ticks":
                # Standardize columns (assuming MT5 names)
                final_df.rename(columns={
                    'time_msc': 'time',
                    'flags': 'flags',
                    'bid': 'bid',
                    'ask': 'ask',
                    'last': 'last',
                    'volume': 'volume',
                }, inplace=True, errors='ignore') # Added errors='ignore'
            else:
                final_df.rename(columns={
                    'tick_volume': 'volume', # Use tick_volume as 'volume'
                }, inplace=True, errors='ignore') # Added errors='ignore'


            # Ensure time column is the primary datetime column and drop time column if exist
            if 'time' in final_df.columns and time_column != 'time':
                 final_df = final_df.drop('time', axis=1)

            # Select desired columns (ensure time_column is first)
            required_columns = [time_column, 'open', 'high', 'low', 'close', 'volume']
            # Keep only columns that actually exist in the dataframe
            final_df = final_df[[col for col in required_columns if col in final_df.columns]]

            # Remove duplicates based on timestamp, keeping the latest entry
            initial_rows = len(final_df)
            final_df = final_df.drop_duplicates(subset=[time_column], keep='last')
            if initial_rows > len(final_df):
                print(f"  Removed {initial_rows - len(final_df)} duplicate rows based on '{time_column}'.")

            # Sort by timestamp
            final_df = final_df.sort_values(by=time_column)

            # Remove the last row *if* it represents the current, incomplete bar.
            if not final_df.empty:
                 # Check if the last bar's time is too close to the script end time
                 # A simple heuristic: if the last bar's time is after the loop's end_dt minus one interval, it might be incomplete.
                 # Or just always drop the last row after bulk download.
                 print("Removing potentially incomplete last bar.")
                 final_df = final_df.iloc[:-1]


        # Apply max rows limit if specified (same as before)
        if download_max_rows and len(final_df) > download_max_rows:
            print(f"Applying download_max_rows limit: {download_max_rows}")
            final_df = final_df.tail(download_max_rows)

        # Final check if DataFrame is valid before saving (same as before)
        if final_df.empty:
             print(f"Final dataframe for {quote} is empty after processing. Skipping save.")
             continue

        # Reset index before saving (same as before)
        final_df = final_df.reset_index(drop=True)

        # --- Save Data (same as before) ---

        try:
            if data_type == "ticks":
                final_df = final_df.drop(['time'], axis=1)

            print(f"Saving {len(final_df)} rows to {file_name}...")
            final_df.to_csv(file_name, index=False, date_format='%Y-%m-%dT%H:%M:%SZ')
            print(f"Finished saving '{quote}'.")
            processed_symbols.append(quote)
        except Exception as e:
            print(f"Error saving file {file_name}: {e}")


    # --- Shutdown MT5 (same as before) ---
    print("\nShutting down MetaTrader 5 connection...")
    mt5.shutdown()

    elapsed = datetime.now() - script_start_time
    print(f"\nFinished downloading data for symbols: {', '.join(processed_symbols) if processed_symbols else 'None'}")
    print(f"Total time: {str(elapsed).split('.')[0]}")


if __name__ == '__main__':
    data_sources = [
        {"symbol": "EURUSD", "type": "klines"},
        {"symbol": "GBPUSD", "type": "klines"}
    ]
    scrape_and_save_candles(
        data_sources,
        mt5_account_id="",
        mt5_password="",
        mt5_server="",  
        timeframe="15min",
        since="2011-01-01T00:00:00Z", 
        until="2024-01-01T23:59:59Z",
    )

