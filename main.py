import os
import gc
import yaml
import sys
import socket
import re
import pandas as pd
from modules.csv_loader import load_subject
from helpers.definitions import Sensor
from modules.filter import run_data_cleansing
from helpers.logger import logger
from modules.export import export_data
from machine_learning.ml_main import do_ml
from modules.relabel import relabel

preprocessing = False
machine_learning = True


def main(config: dict, settings: dict) -> int:
    """
    Function to run the entire preprocessing pipeline, from data loading to cleaning to relabeling etc.
    :param settings: dict containing study wide settings
    :param config: dict containing configuration information, e.g. folders, filenames or other settings
    :return: int: Exit code
    """
    if preprocessing:
        for subject in settings["all_subjects"]:
            export_subfolder = config.get("export_subfolder")
            if not (os.path.isdir(export_subfolder)):
                os.mkdir(export_subfolder)
            if os.path.isfile(export_subfolder + "exported.txt"):
                with open(export_subfolder + "exported.txt", "r") as f:
                    out = False
                    for line in f:
                        if line.strip() == subject:
                            out = True
                if out:
                    continue
            else:
                with open(export_subfolder + "exported.txt", "w") as f:
                    pass
            logger.info(f"########## Starting to run on subject {subject} ##########")
            logger.info(f"##### Loading subject {subject} #####")
            recordings_list = load_subject(subject, config, settings)
            logger.info(f"##### Cleaning subject {subject} #####")
            cleaned_data = run_data_cleansing(recordings_list, subject, config, Sensor.ACCELEROMETER, settings)
            labeled_data = relabel(cleaned_data, config, settings, subject)
            logger.info(f"##### Exporting subject {subject} #####")
            export_data(labeled_data, config, settings, subject)
            logger.info(f"########## Finished running on subject {subject} ##########")
            del recordings_list, cleaned_data, labeled_data  # hopefully fix memory-caused sigkill...
            gc.collect()

        logger.info("Finished running prepocessing")

    if machine_learning:
        folder_path = config.get("export_subfolder")
        pattern = r'OCDetect_(\d+)'

        dataframes = {}

        subject_numbers = [12]

        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                match = re.search(pattern, file_name)
                if match:
                    subject_number = int(match.group(1))

                    if subject_number in subject_numbers:
                        file_path = os.path.join(folder_path, file_name)
                        df = pd.read_csv(file_path, )

                        if subject_number in dataframes:
                            dataframes[subject_number].append(df)
                        else:
                            dataframes[subject_number] = [df]

        #data = pd.DataFrame(dataframes)
        subjects = dataframes.keys()
        data = list(dataframes.values())

        do_ml(data, subjects, config, settings)

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
