import os
import gc

import sklearn.model_selection
import yaml
import sys
import socket
from data_preparation.prepare import prepare_data, get_data_path_variables, load_data_preparation_settings
from misc.csv_loader import load_subject
from data_cleansing.helpers.definitions import Sensor
from data_cleansing.modules.filter import run_data_cleansing
from misc import logger
from misc.export import export_data
import pandas as pd
import getpass
from machine_learning.ml_main import ml_pipeline
from copy import deepcopy

import threading
import concurrent.futures
from multiprocessing import Manager, Lock


def main(config: dict, settings: dict) -> int:

    data_cleansing = settings["data_cleansing"]
    data_preparation = settings["data_preparation"]
    machine_learning = settings["machine_learning"]

    """
    Function to run the entire preprocessing pipeline, from data loading to cleaning to relabeling etc.
    AND/OR run the data cleansing and machine learning pipeline, respectively.
    :param settings: dict containing study wide settings
    :param config: dict containing configuration information, e.g. folders, filenames or other settings
    :return: int: Exit code
    """
    threads = []
    futures = []
    try:
        already_done = pd.read_csv(config["output_folder"] + "prep_params.csv", index_col=False)
    except FileNotFoundError:
        already_done = []  # TODO to be tested for data_cleansing = True
    if data_cleansing:
        with Manager() as manager:
            #subj_loaded = manager.dict()
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as TPE:
                for subject in settings["all_subjects"]:
                    if settings["test_filters"]:
                        grid_definition = {
                            "short_succession_time": settings["short_succession_time"],
                            "magnitude_window_size": settings["magnitude_window_size"],
                            "magnitude_threshold": settings["magnitude_threshold"],
                            "magnitude_overlap": settings["magnitude_overlap"],
                            "min_time_in_s_before_label": settings["min_time_in_s_before_label"]
                        }
                        grid = sklearn.model_selection.ParameterGrid(grid_definition)
                        for i, settings_values in enumerate(grid):
                            cur_settings = deepcopy(settings)
                            already_done_candidates = already_done[already_done.subject==subject].copy()
                            for key, val in settings_values.items():
                                cur_settings[key] = val
                                already_done_candidates = already_done_candidates[already_done_candidates[key]==val]
                            if len(already_done_candidates) != 0:
                                continue
                            cur_settings["repetition"] = i
                            t = TPE.submit(data_cleansing_worker, *(subject, config, cur_settings))#, subj_loaded))
                            futures.append(t)

                    else:  # TODO: repair filters with lists to single value
                        t = threading.Thread(target=data_cleansing_worker, args=(subject, config, settings))
                        threads.append(t)
                        t.start()
                for index, thread in enumerate(threads):
                    gc.collect()
                    thread.join()
                for future in futures:
                    future.result()

            logger.info("Finished running prepocessing")

    # Preparation / Data loading and ML:
    run_deep_learning = settings.get("run_deep_learning")
    run_classic_methods = settings.get("run_classic_methods")
    raw_str = "both" if run_deep_learning and run_classic_methods else ("raw" if run_deep_learning else "features")
    label_type = settings.get("label_type")
    if data_preparation:
        labels, (features, features_raw), users, feature_names = prepare_data(settings, config, raw=raw_str)
    if machine_learning:
        if not data_preparation:
            # load prepared data
            logger.info("Read in prepared data")

            use_filter, use_scaling, resample, balancing_option = load_data_preparation_settings(
                settings)

            window_size, subjects_folder_name, sub_folder_path, export_path, scaling, filtering = get_data_path_variables(
                use_scaling, use_filter, config, settings)

            logger.info(f"Using path: {export_path}{sub_folder_path}")
            logger.info(f"Scaled data: {scaling}; Filtered data: {filtering}")

            logger.info("Reading precalculated windows")

            if run_classic_methods:
                features = pd.read_csv(f"{export_path}{sub_folder_path}/features_{filtering}_{scaling}_{label_type}.csv")
            if run_deep_learning:
                features_raw = pd.read_csv(f"{export_path}{sub_folder_path}/features_{filtering}_{label_type}_raw.csv")
            labels = pd.read_csv(f"{export_path}{sub_folder_path}/labels_{filtering}_{scaling}_{label_type}.csv")
            users = pd.read_csv(f"{export_path}{sub_folder_path}/users_{filtering}_{scaling}_{label_type}.csv")
            feature_names = pd.read_csv(f"{export_path}{sub_folder_path}/feature_names_{filtering}_{scaling}_{label_type}.csv").iloc[:, 0].tolist()
            logger.info("Finished loading precalculated windows")

        seed = settings.get("seed")
        if run_classic_methods:
            ml_pipeline(features, users, labels, feature_names, seed, settings, config, classic=True)
        if run_deep_learning:
            ml_pipeline(features_raw, users, labels, feature_names, seed, settings, config, classic=False)

    return 0


copy_lock = threading.Lock()


def data_cleansing_worker(subject: str, config: dict, settings: dict): # , subjects_loaded: dict):
    subject = str(subject)
    if len(subject) == 1:
        subject = "0" + subject
    export_subfolder = config.get("data_folder_relabeled")
    if not (os.path.isdir(export_subfolder)):
        os.mkdir(export_subfolder)
    if os.path.isfile(export_subfolder + "exported.txt"):
        with open(export_subfolder + "exported.txt", "r") as f:
            out = False
            for line in f:
                if line.strip() == subject:
                    out = True
        if out:
            return
    else:
        with open(export_subfolder + "exported.txt", "w") as f:
            pass
    logger.info(f"########## Starting to run on subject {subject} ##########")
    logger.info(f"##### Loading subject {subject} #####")

    with copy_lock:
        recordings_list = deepcopy(load_subject(subject, config, settings))

    logger.info(f"##### Cleaning subject {subject}:{settings.get('repetition')} #####")
    cleaned_data = run_data_cleansing(recordings_list, subject, config, Sensor.ACCELEROMETER, settings)
    for item in cleaned_data:
        item.clear()
        del item
    del cleaned_data
    gc.collect()

    if not settings["run_export"]:
        for item in recordings_list:
            item.clear()

        for item in cleaned_data:
            item.clear()
        del recordings_list, cleaned_data
        gc.collect()
        return
    # TODO read from a settings file or manual relabeled data or automatic relabeling should be used
    # labeled_data = relabel(cleaned_data, config, settings, subject)
    logger.info(f"##### Exporting subject {subject} #####")
    # export_data(labeled_data, config, settings, subject)
    export_data(cleaned_data, config, settings, subject)
    logger.info(f"########## Finished running on subject {subject} ##########")
    for item in recordings_list:
        item.clear()
    for item in cleaned_data:
        item.clear()
    # for item in labeled_data:
       # item.clear()
    del recordings_list, cleaned_data  #, labeled_data  # hopefully fix memory-caused sigkill...
    gc.collect()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_file_name = sys.argv[1]
        logger.debug(f"Running with config file: '{config_file_name}'")
    else:
        config_file_name = "misc/config/config.yaml"
        logger.debug(f"No config passed via parameters, running with default: '{config_file_name}'")
    username = getpass.getuser()
    try:
        with open(config_file_name, "r") as config_stream:
            configs = yaml.safe_load(config_stream)
            possible_configs = [list(entry.values())[0] for entry in configs if
                             list(entry.values())[0].get("hostname", "") == socket.gethostname() or
                                socket.gethostname() in list(entry.values())[0].get("hostname", "")]
            active_config = [x for x in possible_configs if x.get("username",0) == username][0] if len(possible_configs) > 1 else possible_configs[0]
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
    sys.exit(0)
