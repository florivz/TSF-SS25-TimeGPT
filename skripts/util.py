import pandas as pd
import datetime
import numpy as np

def convert_time_stamp(df: pd.DataFrame, ts_column: str) -> pd.DataFrame:
    if ts_column not in df.columns:
        raise ValueError(f"Column '{ts_column}' not found in DataFrame")
    
    timestamp_s = pd.to_datetime(df[ts_column]).map(datetime.datetime.timestamp)

    day = 24 * 60 * 60

    df['day_sin'] = (np.sin(timestamp_s * (2*np.pi/day))).values
    df['day_cos'] = (np.cos(timestamp_s * (2*np.pi/day))).values
    df.drop(columns=[ts_column], inplace=True)
    return df