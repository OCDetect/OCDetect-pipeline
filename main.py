import os
import gc
import yaml
import sys
import socket
from data_preparation.prepare import prepare_data, get_data_path_variables, load_data_preparation_settings
from misc.csv_loader import load_subject
from data_cleansing.helpers.definitions import Sensor
from data_cleansing.modules.filter import run_data_cleansing
from misc import logger
from misc.export import export_data
from data_cleansing.modules import relabel
import pandas as pd
from machine_learning.ml_main import ml_pipeline

data_cleansing = False
data_preparation = False
machine_learning = True


def main(config: dict, settings: dict) -> int:
    """
    Function to run the entire preprocessing pipeline, from data loading to cleaning to relabeling etc.
    AND/OR run the data cleansing and machine learning pipeline, respectively.
    :param settings: dict containing study wide settings
    :param config: dict containing configuration information, e.g. folders, filenames or other settings
    :return: int: Exit code
    """
    if data_cleansing:
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
    if data_preparation:
        labels, features, users, feature_names = prepare_data(settings, config)
    if machine_learning:
        if not data_preparation:
            # load prepared data
            logger.info("Read in prepared data")

            use_filter, use_scaling, resample, use_undersampling, use_oversampling = load_data_preparation_settings(
                settings)

            window_size, subjects, subjects_folder_name, sub_folder_path, export_path, scaling, filtering = get_data_path_variables(
                use_scaling, use_filter, config, settings)

            logger.info(f"Using path: {export_path}{sub_folder_path}")
            logger.info(f"Scaled data: {scaling}; Filtered data: {filtering}")

            # todo: remove column "unnamed: 0" while writing to file instead of when reading in
            features = pd.read_csv(f"{export_path}{sub_folder_path}/features_{filtering}_{scaling}.csv",
                                   usecols=lambda col: col != "Unnamed: 0")
            labels = pd.read_csv(f"{export_path}{sub_folder_path}/labels_{filtering}_{scaling}.csv",
                                 usecols=lambda col: col != "Unnamed: 0")
            users = pd.read_csv(f"{export_path}{sub_folder_path}/users_{filtering}_{scaling}.csv",
                                usecols=lambda col: col != "Unnamed: 0")
            feature_names = pd.read_csv(f"{export_path}{sub_folder_path}/feature_names_{filtering}_{scaling}.csv",
                                        usecols=lambda col: col != "Unnamed: 0")

        seed = settings.get("seed")
        ml_pipeline(features, users, labels, feature_names, seed, settings, config)

    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_file_name = sys.argv[1]
        logger.debug(f"Running with config file: '{config_file_name}'")
    else:
        config_file_name = "misc/config/config.yaml"
        logger.debug(f"No config passed via parameters, running with default: '{config_file_name}'")
    try:
        with open(config_file_name, "r") as config_stream:
            configs = yaml.safe_load(config_stream)
            active_config = [list(entry.values())[0] for entry in configs if
                             list(entry.values())[0].get("hostname", "") == socket.gethostname()][0]
        with open("misc/config/settings.yaml", "r") as settings_stream:
            # settings = list(yaml.load_all(settings_stream, Loader=yaml.SafeLoader))
            settings = yaml.safe_load(settings_stream)
    except FileNotFoundError:
        logger.error(f"Could not load config file {config_file_name}, exiting...")
        sys.exit(1)
    except IndexError:
        logger.error(f"Hostname {socket.gethostname()} not contained in config file '{config_file_name}', exiting...")
        sys.exit(1)
    main(config=active_config, settings=settings)
