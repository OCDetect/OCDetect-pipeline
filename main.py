import sys
import os
import yaml
import socket
from typing import Union
from modules.csv_loader import load_subject
from helpers.misc import calc_magnitude
from helpers.definitions import Sensor
from modules.filter import run_data_cleansing


def main(config: dict) -> int:
    """
    Function to run the entire preprocessing pipeline, from data loading to cleaning to relabeling etc.
    :param config: dict containing configuration information, e.g. folders, filenames or other settings
    :return: int: Exit code
    """
    subject = "03"
    recordings_list = load_subject(subject, config)
    print(recordings_list[0].datetime)

    cleaned_data = run_data_cleansing(recordings_list, subject, config, Sensor.ACCELEROMETER)

    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_file_name = sys.argv[1]
        print(f"Running with config file: {config_file_name}")
    else:
        config_file_name = "config/config.yaml"
        print(f"No config defined, running with default")
    try:
        with open(config_file_name, "r") as config_stream:
            configs = yaml.safe_load(config_stream)
            active_config = [list(entry.values())[0] for entry in configs if
                             list(entry.values())[0].get("hostname", "") == socket.gethostname()][0]
    except FileNotFoundError:
        print(f"Could not load config file {config_file_name}")
        sys.exit(1)
    except IndexError:
        print(f"Hostname {socket.gethostname()} not contained in config file")
        sys.exit(1)
    main(config=active_config)
