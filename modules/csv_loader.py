import os
import pandas as pd
import datetime
from glob import glob
from typing import List, Dict, Tuple, Union
from pathlib import Path
from helpers.misc import get_metadata, add_timezone_and_summertime, get_file_name_initial_hw
from helpers.logger import logger
from helpers.definitions import IgnoreReason
from tqdm import tqdm
import numpy as np


def get_subject_filelist(subject_id: str, config: dict, settings: dict) -> List[str]:
    filelist_all = glob(config["data_folder"] + "*/*.csv")
    subject_filelist = [f for f in filelist_all if subject_id in Path(f).parent.name]

    n_test = int(settings.get("test_n_recs", 0))
    if n_test > 0:
        subject_filelist = np.random.choice(subject_filelist, n_test, replace=False)
    return subject_filelist


def load_subject(subject_id: str, config: dict, settings: dict)\
        -> Union[List[pd.DataFrame], Tuple[List[pd.DataFrame], dict]]:
    """
    Load a single subject. All recordings are loaded and augmented with relevant information:
        - datetime instead of timestamp, taken from metadata
        - duplicate lines are dropped (only appeared at the end of some corrupted files)
    :param subject_id: The subject to be loaded
    :param config: dict containing configuration information, e.g. folders, filenames or other settings
    :param settings: the global settings
    :return: List with pd.DataFrames, one per recording of the subjects and the position in the list, dates (optional)
    """

    first_hw_csv_name = get_file_name_initial_hw(subject_id, config)

    recordings = []

    subject_filelist = get_subject_filelist(subject_id, config, settings)

    for filename in tqdm(subject_filelist, smoothing=0.5):
        recording_df = load_recording(filename)
        recording = Path(filename).stem
        date, _ = get_metadata(subject_id, recording, config)
        date_base = add_timezone_and_summertime(date)
        recording_df["datetime"] = recording_df.timestamp.apply(
            lambda x: date_base + datetime.timedelta(seconds=x / 1e9))
        recording_df.drop_duplicates(keep=False, inplace=True)
        recording_df.recording = recording

        recording_df['ignore'] = IgnoreReason.DontIgnore
        if recording == first_hw_csv_name:
            first_hw_path = config["data_folder"] + config["first_hw_subfolder"]
            first_hw_csv = load_recording(first_hw_path + "/labels_" + first_hw_csv_name + ".csv", sep=",")
            initial_hw_indices = (recording_df.index >= int(first_hw_csv['start'])) & \
                                 (recording_df.index <= int(first_hw_csv['end']))
            recording_df.loc[initial_hw_indices, 'ignore'] = IgnoreReason.InitialHandWash
            recording_df.loc[1 - initial_hw_indices, 'ignore'] = IgnoreReason.DontIgnore
        recordings.append(recording_df)
    return recordings


def load_recording(filename: str, sep="\t") -> pd.DataFrame:
    filepath = Path(filename)
    rec_df = pd.read_csv(filepath, sep=sep)
    return rec_df


def load_subjects(subjects: List[str], config: dict, settings: dict) -> List[List[pd.DataFrame]]:
    """
    :param config:
    :param subjects: Subject ids to load
    :param settings: dict containing study wide settings
    :return:
    """
    out_list = []
    for subject in subjects:
        out_list.append(load_subject(subject, config, settings))
    return out_list


def load_all_subjects(config: dict, settings: dict) -> Tuple[Dict[str, int], List[List[pd.DataFrame]]]:
    """
    :param config: dict containing configuration information, e.g. folders, filenames or other settings
    :param settings: dict containing study wide settings
    :return: List of Lists with pd.DataFrames. One List per Subject, each containing all recordings of the subject.
    """

    # all_subjects = [x.path[-2:] for x in os.scandir(config["data_folder"]) if x.is_dir() and x.path[-3] == "/"              and x.path[-1].isdigit() and x.path[-2].isdigit()]

    file_names = [f"{config.get('prefix', '')}{ids}" for ids in settings.get("all_subjects")]
    all_subjects = [x.name[-2:] for x in os.scandir(config["data_folder"]) if x.is_dir() and x.name in file_names]

    if len(all_subjects) != len(settings.get("all_subjects")):
        missing_subjects = [s for s in file_names if
                            not any(s in f"{config.get('prefix', '')}{sub}" for sub in all_subjects)]
        logger.warning(f"not all subjects' data could be found, missing subjects: {missing_subjects}")
    else:
        logger.debug(f"found all {len(all_subjects)} subjects")

    test_subs = settings.get("test_n_subs", 0)
    if test_subs > 0:
        all_subjects = np.random.choice(all_subjects, test_subs, replace=False)
        logger.debug(f"running on subjects: {all_subjects}")

    out_list = load_subjects(all_subjects, config, settings)
    list_map = {subject: i for i, subject in enumerate(all_subjects)}
    return list_map, out_list


def load_all_labels(config: dict, settings: dict) -> pd.DataFrame:
    """
    Function to load all labels in the files where only the labels are stored ("allEvaluations").
    :param config: dict containing configuration information, e.g. folders, filenames or other settings
    :param settings: dict containing study wide settings
    :return: pandas.Dataframe, containing all evaluations collected in the study
    """
    evals_folder = config.get("evals_subfolder", None)
    if evals_folder is None:
        logger.error("Evals_subfolder not set in config")
        return
    filelist_all = glob(config["data_folder"] + evals_folder + "*/*.csv")
    evals = []
    for filename in filelist_all:
        if os.path.getsize(filename) == 0:
            continue
        try:
            df = pd.read_csv(filename, sep="\t", names=["timestamp", "user yes/no", "compulsive", "tense", "urge"],
                             header=None)
            path = Path(filename)
            filename = path.name
            subject = path.parent.name
            df["recording"] = filename[:filename.index('_')]
            df["subject"] = subject[:2]
            evals.append(df)
        except pd.errors.ParserError as e:
            pass  # logger.error(filename + ": " + str(e))
    evals = pd.concat(evals)
    return evals[evals["user yes/no"] == 1]


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
        csv = load_recording(config["data_folder"] + config.get("prefix", "")
                             + subject + "/" + first_hw_csv_name + ".csv")
        first_hw_csv = load_recording(first_hw_path + "/labels_" + first_hw_csv_name + ".csv", sep=",")
        # since first hw csv is a csv with only one line, it comes as a series, so we need .values[0]
        begin_ts = csv.iloc[first_hw_csv['start'].values[0]]['timestamp']
        end_ts = csv.iloc[first_hw_csv['end'].values[0]]['timestamp']
        return int((end_ts / 1000000000) - (begin_ts / 1000000000))

    # take the average hand washing time if no initial recording was found
    return avg_hand_wash_time
