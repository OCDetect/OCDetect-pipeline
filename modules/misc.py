import datetime
from typing import Tuple, Union
import json
from pathlib import Path

def initial_handwash_time():
    """
    :return:
    """
    return 39


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
