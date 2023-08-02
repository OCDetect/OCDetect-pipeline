import os
import gc
import yaml
import sys
import socket
from data_cleansing.modules import load_subject
from data_cleansing.helpers.definitions import Sensor
from data_cleansing.modules import run_data_cleansing
from misc import logger
from misc.export import export_data
from ml_main import do_ml
from data_cleansing.modules import relabel

preprocessing = False
machine_learning = True


def main(config: dict, settings: dict) -> int:
    """
    Function to run the entire preprocessing pipeline, from data loading to cleaning to relabeling etc.
    AND/OR run the data cleansing and machine learning pipeline, respectively.
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
        do_ml(config, settings, prepare_data=False, machine_learning=True)

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
