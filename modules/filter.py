import numpy as np
import pandas as pd
from helpers.definitions import Sensor
from helpers.logger import logger
from helpers.misc import get_initial_hw_datetime, initial_handwash_time, calc_magnitude
from typing import List
from tqdm import tqdm


def run_data_cleansing(recordings_list: List[pd.DataFrame], subject: str, config: dict, sensor: Sensor) -> List[pd.DataFrame]:
    """

    :param sensor: which sensor to be used for calculating idle regions (gyroscope or accelerometer)
    :param recordings_list: all original recordings from one subject
    :param subject: the subject from the recordings
    :param config: the loaded config
    :return: a list of DataFrames with the recordings that passed the filter rules
    """

    cleaned_recordings_list = []
    filtered_out_files = 0

    initial_hw_time = initial_handwash_time(subject, config)
    for recording in recordings_list:
        # Filter out complete recordings

        # 1. check if file has content at all
        if not check_file_corrupt(recording):
            cleaned_recordings_list.append(recording)
        else:
            filtered_out_files += 1

        # 2. check if recording time is smaller the person specific initial hand washing time
        if not check_insufficient_file_length(recording, initial_hw_time):
            cleaned_recordings_list.append(recording)
        else:
            filtered_out_files += 1

        # 3. check if recording date is before initial hw recording
        if not check_recording_before_initial_hw(recording, subject, config):
            cleaned_recordings_list.append(recording)
        else:
            filtered_out_files += 1

        # 4. check if has no movement at all (delete when remaining windows are smaller than initial hw time)
        recording = calc_magnitude(recording, sensor)
        recording = calc_idle_time(recording, sensor)
        if not check_insufficient_remaining_data_points(recording, initial_hw_time):
            cleaned_recordings_list.append(recording)
        else:
            filtered_out_files += 1

    percentage_filtered_out = (filtered_out_files * 100)/len(recordings_list)
    logger.info(f"Complete recordings filtered out: {filtered_out_files} ({percentage_filtered_out:.2f}%)")

    return cleaned_recordings_list


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
        logger.logerror("please calculate magnitude before")
        return data

    data["idle"] = np.nan
    stride = int(window_size * overlap)

    for i in tqdm(range(0, len(data) - window_size + 1, stride)):
        cur_win = data.iloc[i:i + window_size]
        std = cur_win[f"mag {sensor.value}"].std()
        if std <= threshold:
            data.loc[i:i + window_size, "idle"] = 1.0

    return data


def check_file_corrupt(data: pd.DataFrame) -> bool:
    """
    Checks if file is empty or contains the header only
    :param data: the dataframe from the csv recording file
    :return: true if corrupt and to be ignored, false otherwise
    """

    return data.empty


def check_insufficient_remaining_data_points(recording_w_idle: pd.DataFrame, initial_hw_time: int) -> bool:
    """
    Checks if the amount of data points that are labelled as not idle has still enough information.
    This is the case if at least a recording remains that is as long as the initial hand washing.
    :param recording_w_idle: the recording with calculated idle regions
    :param initial_hw_time: the time in seconds for the initial hand washing recording
    :return: true if recording has enough non-idle regions, false otherwise
    """
    amount = len(recording_w_idle[recording_w_idle["idle"] == 1.0])
    # sampling frequency == 50 Hz -> amount of non-idle data points divided by 50 equals remaining time
    return int(amount / 50) > initial_hw_time


def check_insufficient_file_length(data: pd.DataFrame, initial_hw_time: int) -> bool:
    """
    Checks if file length is too small to contain relevant information. Since we focus on hand washing and do not want
    to miss that the file length has to be at least as long as the inital hand washing time.
    :param data: the dataframe from the csv recording file
    :param initial_hw_time: the time in seconds for the initial hand washing recording
    :return: true if file is too short, false otherwise
    """

    return int(data.iloc[-1]["timestamp"] / 1000000000) < initial_hw_time


def check_recording_before_initial_hw(data: pd.DataFrame, subject: str, config: dict) -> bool:
    """
    Checks if the given recording happened before the initial lab session.
    :param data: the dataframe from the csv recording file
    :param subject: the subject from the recording
    :param config: the loaded config file
    :return: true if recording was before initial lab session, false otherwise
    """

    return data.iloc[0]["datetime"].date() < get_initial_hw_datetime(subject, config).date()
