import numpy as np
import pandas as pd
from helpers.definitions import Sensor, IgnoreReason
from helpers.logger import logger
from typing import List
from tqdm import tqdm
from visualizations.line_plotter import plot_idle_regions
from helpers.misc import get_initial_hw_datetime, calc_magnitude
from modules.csv_loader import initial_handwash_time


def run_data_cleansing(recordings_list: List[pd.DataFrame], subject: str, config: dict, sensor: Sensor,
                       settings: dict) -> List[pd.DataFrame]:
    """

    :param sensor: which sensor to be used for calculating idle regions (gyroscope or accelerometer)
    :param recordings_list: all original recordings from one subject
    :param subject: the subject from the recordings
    :param config: the loaded config
    :param settings: the study wide settings dict
    :return: a list of DataFrames with the recordings that passed the filter rules
    """

    cleaned_recordings_list = []
    filtered_out_files = 0
    overall_idle_regions = 0
    overall_regions = 0
    modified_labels = 0
    removed_labels = 0

    initial_hw_time = initial_handwash_time(subject, config)
    for counter, recording in enumerate(recordings_list):

        ################################
        # Filter out complete recordings
        ################################

        # 1. check if file has content at all
        if check_file_corrupt(recording):
            filtered_out_files += 1
            logger.info("File has no content.")
            continue

        # 2. check if recording time is smaller the person specific initial hand washing time
        if check_insufficient_file_length(recording, initial_hw_time):
            filtered_out_files += 1
            logger.info("File is too short.")
            continue

        # 3. check if recording date is before initial hw recording
        if check_recording_before_initial_hw(recording, subject, config):
            filtered_out_files += 1
            logger.info("Recording date is before initial lab session.")
            continue

        # 4. check if has no movement at all (delete when remaining windows are smaller than initial hw time)
        recording = calc_magnitude(recording, sensor)
        recording = calc_idle_time(recording, sensor, settings)
        if check_insufficient_remaining_data_points(recording, initial_hw_time):
            filtered_out_files += 1
            logger.info("File has too little movement.")
            continue

        ##########################################################
        # Filter regions in recordings by setting an "ignore" flag
        ##########################################################

        # 1. ignore regions that have no movement
        recording = set_ignore_no_movement(recording)

        # 2. when recording includes initial hw, ignore regions that were under supervision
        # this is already handled when data is read in because this is the only time
        # we still have the connection between data and file

        # 3. label was set too early in file that was cannot be hand washing before
        recording = check_for_too_early_label(recording, 5)

        recording_ignore_regions = len(recording[recording["ignore"] == IgnoreReason.DontIgnore])
        overall_idle_regions += recording_ignore_regions

        percentage_ignore_regions = (recording_ignore_regions*100)/len(recording)
        overall_regions += len(recording)

        logger.info(f"Percentage of the file to be ignored: {percentage_ignore_regions:.2f}%)")

        plot_idle_regions(config, recording, Sensor.ACCELEROMETER, title=f"percentage of ignored regions: {percentage_ignore_regions:.2f}%", save_fig=True, fig_name=f"{subject}_{counter}")
        # X. find and handle labels placed by the subjects in short succession TODO
        recording = short_succession(recording, subject, config, settings)

    percentage_filtered_out = (filtered_out_files * 100) / len(recordings_list)
    percentage_ignore_overall_regions = (overall_idle_regions*100)/overall_regions

    logger.info(f"###############################################################")
    logger.info(f"Data cleaning stats for subject {subject}")
    logger.info(f"Complete recordings filtered out: {filtered_out_files} out of {len(recordings_list)} ({percentage_filtered_out:.2f}%)")
    logger.info(f"Overall regions marked as to be ignored: {percentage_ignore_overall_regions:.2f}%")
    logger.info(f"###############################################################")

    return cleaned_recordings_list


# TODO add tests for the filter methods
def check_for_too_early_label(data: pd.DataFrame, min_time: int) -> pd.DataFrame:
    frequency = 50
    idx = min_time * frequency
    label_idx = data.loc[:idx, "user yes/no"][data.loc[:idx, "user yes/no"] == 1.0].index

    if len(label_idx) > 0:
        data.loc[:label_idx[-1], 'ignore'] = IgnoreReason.TooEarlyInRecording

    return data


# TODO add tests for the filter methods
def set_ignore_no_movement(data: pd.DataFrame) -> pd.DataFrame:
    data.loc[data['idle'] == 1.0, 'ignore'] = IgnoreReason.NoMovement
    return data


def calc_idle_time(data: pd.DataFrame, sensor: Sensor, settings: dict) -> pd.DataFrame:
    """
    Calculates idle regions in a dataframe with windowing based on the magnitude values and a threshold by using std.
    Adds a column "idle" to the dataframe with NaNs for non-idle regions and 1.0 for idle regions
    :param settings: the study wide settings dict
    :param data: the dataframe to calculate idle time
    :param sensor: the sensor (accelerometer or gyroscope) to use for calculating the idle time
    :return: the dataframe with an additional column for idle regions
    """

    if f"mag {sensor.value}" not in data.columns:
        logger.logerror("please calculate magnitude before")
        return data

    window_size = settings.get("magnitude_window_size", 500)
    threshold = settings.get("magnitude_threshold", 0.2)
    overlap = settings.get("magnitude_overlap", 0.5)

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
    return int(amount / 50) < initial_hw_time


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


def short_succession(data: pd.DataFrame, subject: str, config: dict, settings: dict, return_counts: bool = False)\
        -> pd.DataFrame:
    """
    Finds (and corrects, TODO)
    labels that occur in short succession of each other. Also checks if the feedback values changed
    :param data: pd.DataFrame of one recording's labels only
    :param subject: the subject from the recording
    :param config: dict containing the configuration for this software (file paths etc.)
    :param settings: dict containing study wide settings
    :param return_counts: whether to return the counts of the duplicated labels or not.
    :return: stats, cleaned recording TODO

    """
    short_succession_time = settings.get("short_succession_time", 0)

    if len(data) <= 1 or short_succession_time <= 0:  # nothing to do here
        if return_counts:
            return data, 0
        return data
    counts = [0, 0, 0, 0]  # same, comp to normal, normal to comp
    try:
        df_copy = data.drop(["recording", "subject"], axis=1).reset_index().diff().reset_index()
    except KeyError:
        df_copy = data[["compulsive", "timestamp"]].diff().reset_index()
    for index, row in df_copy.iterrows():
        timediff = row.timestamp / 1e9  # in seconds
        if timediff < short_succession_time:
            if row.compulsive == -1:  # compulsive to routine
                counts[1] += 1
                df_copy.loc[index, 'ignore'] = IgnoreReason.RepetitionCompToRoutine
            elif row.compulsive == 1:  # routine to compulsive
                counts[2] += 1
                df_copy.loc[index, 'ignore'] = IgnoreReason.RepetitionRoutineToComp
            else:  # repetition of the previous label (comp->comp or routine->routine)
                counts[0] += 1
                df_copy.loc[index, 'ignore'] = IgnoreReason.RepetitionSame
    if return_counts:
        return data, counts
    return data
