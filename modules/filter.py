import numpy as np
import pandas as pd
from helpers.definitions import Sensor


def calc_idle_time(data: pd.DataFrame, sensor: Sensor, threshold=0.5, window_size=50, overlap=0.5) -> pd.DataFrame:
    """
    Calculates idle regions in a dataframe with windowing based on the magnitude values and a threshold by using std.
    Adds a column "idle" to the dataframe with NaNs for non-idle regions and 1.0 for idle regions
    :param overlap: value between 0 and 1 for percentage of window overlap, default 0.5 for 50% overlap
    :param window_size: amount of samples, default 50, (1s because of 50Hz)
    :param threshold: value for which regions are marked as idle based on the std
    :param data: the dataframe to calculate idle time
    :param sensor: the sensor (accelerometer or gyroscope) to use for calculating the idle time
    :return: the dataframe with an additional column for idle regions
    """

    if f"mag {sensor.value}" not in data.columns:
        print("please calculate idle time before") # TODO add logger for giving feedback instead of prints
        return data

    data["idle"] = np.nan
    stride = int(window_size * overlap)

    for i in range(0, len(data) - window_size + 1, stride):
        cur_win = data.iloc[i:i+window_size]
        std = cur_win[f"{sensor.value} acc"].std()
        if std <= threshold:
            data.loc[i:i+window_size, "idle"] = 1.0

    return data


def check_file_corrupt(data: pd.DataFrame) -> bool:
    """
    Checks if data is empty or contains the header only
    :param data: the dataframe from the csv recording file
    :return: true if corrupt and to be ignored, false otherwise
    """

    return data.empty
