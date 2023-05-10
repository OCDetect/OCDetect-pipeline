import sys
import os
import yaml
import socket
import logging
from typing import Union
from modules.csv_loader import load_subject, load_all_subjects
from helpers.misc import calc_magnitude
from helpers.definitions import Sensor
from modules.filter import run_data_cleansing
# import helpers.logger  # the import statement is enough to initialize the logger KK: I dont think this import statement is needed at all...
import numpy as np
from visualizations.line_plotter import plot_3_axis


def main(config: dict, settings: dict) -> int:
    """
    Function to run the entire preprocessing pipeline, from data loading to cleaning to relabeling etc.
    :param settings:
    :param config: dict containing configuration information, e.g. folders, filenames or other settings
    :return: int: Exit code
    """

    subject_map, subject_recordings = load_all_subjects(config, settings)
    subject = np.random.choice(list(subject_map.keys()))

    recordings_list = subject_recordings[subject_map[subject]]

    # cleaned_data = run_data_cleansing(recordings_list, subject, config, Sensor.ACCELEROMETER)
    plot_3_axis(config, recordings_list[0], Sensor.ACCELEROMETER, start_idx=2000, end_idx=4000, save_fig=True)
    logging.info("Finished running prepocessing")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_file_name = sys.argv[1]
        logging.debug(f"Running with config file: '{config_file_name}'")
    else:
        config_file_name = "config/config.yaml"
        logging.debug(f"No config passed via parameters, running with default: '{config_file_name}'")
    try:
        with open(config_file_name, "r") as config_stream:
            configs = yaml.safe_load(config_stream)
            active_config = [list(entry.values())[0] for entry in configs if
                             list(entry.values())[0].get("hostname", "") == socket.gethostname()][0]
        with open("config/settings.yaml", "r") as settings_stream:
            # settings = list(yaml.load_all(settings_stream, Loader=yaml.SafeLoader))
            settings = yaml.safe_load(settings_stream)
    except FileNotFoundError:
        logging.error(f"Could not load config file {config_file_name}, exiting...")
        sys.exit(1)
    except IndexError:
        logging.error(f"Hostname {socket.gethostname()} not contained in config file '{config_file_name}', exiting...")
        sys.exit(1)
    main(config=active_config, settings=settings)
