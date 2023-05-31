import datetime
import logging
from typing import Tuple, Union
import json
from pathlib import Path
import os
import pandas as pd
import numpy as np
from helpers.definitions import Sensor
import modules.csv_loader as loader


def get_file_name_initial_hw(subject: str, config: dict) -> str:
    """
    Returns the file name for the CSV file containing the initial hand washing activity in the lab by taking the defined
    start and end index from the first_hw_subfolder and searching for the file name string in the subject's
    csv recordings file lists.
    :param subject: The subject to be loaded
    :param config: dict containing configuration information, e.g. folders, filenames or other settings
    :return: the file name if initial hand washing recording was found, empty string otherwise
    """
    first_hw_path = config["data_folder"] + config["first_hw_subfolder"]
    first_hw_csv_names = [first_hw[7:] for first_hw in os.listdir(first_hw_path) if first_hw.endswith(".csv")]

    csv_name_from_subject = [f for f in os.listdir(config["data_folder"] + config.get("prefix", "") + subject) if
                             f.endswith('.csv')]
    for first_hw_csv_name in first_hw_csv_names:
        if first_hw_csv_name in csv_name_from_subject:
            return first_hw_csv_name[:-4]

    return ""


def initial_handwash_time(subject: str, config: dict) -> int:
    """
    Calculates the initial hand washing time from the observed recording in the lab.
    If no initial hand washing was recorded, the mean time of 39s from all other
    initial hw recordings is returned.
    :param subject: The subject to be loaded
    :param config: dict containing configuration information, e.g. folders, filenames or other settings
    :return: the initial hand washing time in seconds if given, 39 as default otherwise
    """
    avg_hand_wash_time = 39
    first_hw_path = config["data_folder"] + config["first_hw_subfolder"]
    first_hw_csv_name = get_file_name_initial_hw(subject, config)

    if len(first_hw_csv_name) > 0:
        csv = loader.load_recording(config["data_folder"] + config.get("prefix", "")
                                    + subject + "/" + first_hw_csv_name + ".csv")
        first_hw_csv = loader.load_recording(first_hw_path + "/labels_" + first_hw_csv_name + ".csv", sep=",")
        # since first hw csv is a csv with only one line, it comes as a series, so we need .values[0]
        begin_ts = csv.iloc[first_hw_csv['start'].values[0]]['timestamp']
        end_ts = csv.iloc[first_hw_csv['end'].values[0]]['timestamp']
        return int((end_ts / 1000000000) - (begin_ts / 1000000000))

    # take the average hand washing time if no initial recording was found
    return avg_hand_wash_time


def get_initial_hw_datetime(subject: str, config: dict) -> datetime.datetime:
    """
    Calculates the date for the initial hand washing recording session in the lab.
    :param subject: The subject to be loaded
    :param config: dict containing configuration information, e.g. folders, filenames or other settings
    :return: the datetime if initial hand washing recording was found, None otherwise
    """
    first_hw_csv_name = get_file_name_initial_hw(subject, config)
    if first_hw_csv_name:
        date, _ = get_metadata(subject, first_hw_csv_name, config)
        return date

    return None


def is_summertime_in2022(date: datetime.datetime, verbose: bool = False) -> bool:
    """
    Checks if a date between late 2021 and late 2022 falls into the period of CEST
    :param date: date to check
    :param verbose: whether to print the result
    :return: true if the date is in the CEST, false otherwise
    """
    begin = datetime.datetime(year=2022, month=3, day=27, hour=2)
    end = datetime.datetime(year=2022, month=10, day=30, hour=2)
    if verbose:
        print("Is summertime:", begin < date < end)
    return begin < date < end


def add_timezone_and_summertime(date: datetime.datetime) -> datetime.datetime:
    """
    Adds one or two hours to a UTC date, in order to adjust it to the local timezone (+ summertime if applicable)
    :param date: the date to change
    :return: the changed date
    """
    return date + datetime.timedelta(hours=(2 if is_summertime_in2022(date) else 1))


def get_metadata(subject: str, recording: str, config: dict) -> Tuple[Union[datetime.datetime, dict]]:
    """
    Loads the metadata json file of a recording
    :param subject: -
    :param recording: -
    :param config: the config dict, that contains the values
    :return: the date of the recording and the entire parsed json file as dict
    """
    filename = config["data_folder"] + config["meta_subfolder"] + subject + f"/{recording}.json"
    try:
        with open(filename, "r") as json_file:
            rec_name = Path(filename).name[:-5]
            subject = Path(filename).parent.name
            json_vals = json.load(json_file)
            date = datetime.datetime.strptime(json_vals["date"][:-5], "%Y-%m-%dT%H:%M:%S")
    except FileNotFoundError:
        logging.error("Metadata-file not found: ", filename)

    return date, json_vals


def calc_magnitude(data: pd.DataFrame, sensor: Sensor) -> pd.DataFrame:
    """
    Calculates the magnitude for a given sensor
    :param data: the dataframe with the sensor data to calculate magnitude for
    :param sensor: the sensor (accelerometer or gyroscope) to calculate the magnitude
    :return: the dataframe with an additional column for the magnitude
    """
    mag = np.sqrt(data[f"{sensor.value} x"] ** 2 + data[f"{sensor.value} y"] ** 2 + data[f"{sensor.value} z"] ** 2)
    data[f"mag {sensor.value}"] = mag

    return data
