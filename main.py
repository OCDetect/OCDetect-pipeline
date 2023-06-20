import sys
import os
import yaml
import socket
from typing import Union
from modules.csv_loader import load_subject, load_all_subjects, load_recording
from helpers.misc import calc_magnitude
from helpers.definitions import Sensor
from modules.filter import run_data_cleansing
from modules.relabel import relabel
from helpers.logger import logger  # the import statement is enough to initialize the logger KK: I dont think this import statement is needed at all... RB: Well yes it is, but only if i also commit the logger.py file...
from modules.export import export_data
import numpy as np
from visualizations.line_plotter import plot_3_axis, plot_magnitude_around_label


def main(config: dict, settings: dict) -> int:
    """
    Function to run the entire preprocessing pipeline, from data loading to cleaning to relabeling etc.
    :param settings: dict containing study wide settings
    :param config: dict containing configuration information, e.g. folders, filenames or other settings
    :return: int: Exit code
    """

    # subject_map, subject_recordings = load_all_subjects(config, settings)
    # subject = np.random.choice(list(subject_map.keys()))

    # recordings_list = subject_recordings[subject_map[subject]]
    recordings_list = load_subject("01", config, settings)
    cleaned_data = recordings_list  # = run_data_cleansing(recordings_list, "01", config, Sensor.ACCELEROMETER, settings)
    labeled_data = relabel(cleaned_data, config, settings, "01")
    export_data(labeled_data, config, settings, "01")

    # for i, recording in enumerate(subject_recordings):
    # cleaned_data = run_data_cleansing(recording, subject_map[i], config, Sensor.ACCELEROMETER, settings)

    # Test plotting stuff
    # plot_3_axis(config, recordings_list[0], Sensor.ACCELEROMETER, start_idx=2000, end_idx=4000, save_fig=True)
    # df = load_recording(f"{config['data_folder']}/OCDetect_12/relabeled_user-one/21.csv", ",")
    # plot_magnitude_around_label(config, df, Sensor.ACCELEROMETER, 565673, 90, 10)

    logger.info("Finished running prepocessing")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_file_name = sys.argv[1]
        logger.debug(f"Running with config file: '{config_file_name}'")
    else:
        config_file_name = "config/config.yaml"
        logger.debug(f"No config passed via parameters, running with default: '{config_file_name}'")
    try:
        with open(config_file_name, "r") as config_stream:
            configs = yaml.safe_load(config_stream)
            active_config = [list(entry.values())[0] for entry in configs if
                             list(entry.values())[0].get("hostname", "") == socket.gethostname()][0]
        with open("config/settings.yaml", "r") as settings_stream:
            # settings = list(yaml.load_all(settings_stream, Loader=yaml.SafeLoader))
            settings = yaml.safe_load(settings_stream)
    except FileNotFoundError:
        logger.error(f"Could not load config file {config_file_name}, exiting...")
        sys.exit(1)
    except IndexError:
        logger.error(f"Hostname {socket.gethostname()} not contained in config file '{config_file_name}', exiting...")
        sys.exit(1)
    main(config=active_config, settings=settings)
