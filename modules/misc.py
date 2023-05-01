import datetime
from typing import Tuple, Union
import json
from pathlib import Path
import os
import pandas as pd


def initial_handwash_time(subject: str, config: dict) -> int:
    """
    Calculates the initial hand washing time from the observed recording in the lab by taking the defined
    start and end index from the first_hw_subfolder and searching for the file name string in the subject's
    csv recordings file lists. If no initial hand washing was recorded, the mean time of 39s from all other
    initial hw recordings is returned.
    :param subject_id: The subject to be loaded
    :param config: dict containing configuration information, e.g. folders, filenames or other settings
    :return: the initial hand washing time in seconds if given, 39 as default otherwise
    """
    avg_hand_wash_time = 39

    first_hw_path = config["data_folder"] + config["first_hw_subfolder"]
    first_hw_csv_names = [first_hw[7:] for first_hw in os.listdir(first_hw_path) if first_hw.endswith(".csv")]

    csv_name_from_subject = [f for f in os.listdir(config["data_folder"] + config["prefix"] + subject) if f.endswith('.csv')]

    for first_hw_csv_name in first_hw_csv_names:
        if first_hw_csv_name in csv_name_from_subject:
            csv = pd.read_csv(config["data_folder"] + config["prefix"] + subject + "/" + first_hw_csv_name, sep="\t")
            first_hw_csv = pd.read_csv(first_hw_path + "/labels_" + first_hw_csv_name)
            # since first hw csv is a csv with only one line, it comes as a series, so we need .values[0]
            begin_ts = csv.iloc[first_hw_csv['start'].values[0]]['timestamp']
            end_ts = csv.iloc[first_hw_csv['end'].values[0]]['timestamp']
            return int((end_ts / 1000000000) - (begin_ts / 1000000000))

    # take the average hand washing time if no initial recording was found
    return avg_hand_wash_time


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
        print("Metadata-file not found: ", filename)

    return date, json_vals
