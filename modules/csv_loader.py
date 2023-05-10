import os
import pandas as pd
import datetime
import logging
from glob import glob
from typing import List, Dict, Tuple
from pathlib import Path
from helpers.misc import get_metadata, add_timezone_and_summertime
from tqdm import tqdm
import numpy as np


def get_subject_filelist(subject_id: str, config: dict, settings: dict) -> List[str]:
    filelist_all = glob(config["data_folder"] + "*/*.csv")
    subject_filelist = [f for f in filelist_all if subject_id in Path(f).parent.name]

    n_test = int(settings.get("test_n_recs", 0))
    if n_test > 0:
        subject_filelist = np.random.choice(subject_filelist, n_test, replace=False)
    return subject_filelist


def load_subject(subject_id: str, config: dict, settings: dict) -> List[pd.DataFrame]:
    """
    Load a single subject. All recordings are loaded and augmented with relevant information:
        - datetime instead of timestamp, taken from metadata
        - duplicate lines are dropped (only appeared at the end of some corrupted files)
    :param subject_id: The subject to be loaded
    :param config: dict containing configuration information, e.g. folders, filenames or other settings
    :return: List with pd.DataFrames, one per recording of the subjects.
    """
    recordings = []

    subject_filelist = get_subject_filelist(subject_id, config, settings)

    for filename in tqdm(subject_filelist, smoothing=0.5):
        recording_df = load_recording(filename)
        recording = Path(filename).stem
        date, _ = get_metadata(subject_id, recording, config)
        date_base = add_timezone_and_summertime(date)
        recording_df["datetime"] = recording_df.timestamp.apply(lambda x: date_base + datetime.timedelta(seconds=x/1e9))
        recording_df.drop_duplicates(keep=False, inplace=True)
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
    :return:
    """
    out_list = []
    for subject in subjects:
        out_list.append(load_subject(subject, config, settings))
    return out_list


def load_all_subjects(config: dict, settings: dict) -> Tuple[Dict[str, int], List[List[pd.DataFrame]]]:
    """
    :param config: dict containing configuration information, e.g. folders, filenames or other settings
    :return: List of Lists with pd.DataFrames. One List per Subject, each containing all recordings of the subject.
    """

    # all_subjects = [x.path[-2:] for x in os.scandir(config["data_folder"]) if x.is_dir() and x.path[-3] == "/"              and x.path[-1].isdigit() and x.path[-2].isdigit()]

    file_names = [f"{config.get('prefix', '')}{ids}" for ids in settings.get("all_subjects")]
    all_subjects = [x.name[-2:] for x in os.scandir(config["data_folder"]) if x.is_dir() and x.name in file_names]

    if len(all_subjects) != len(settings.get("all_subjects")):
        missing_subjects = [s for s in file_names if not any(s in f"{config.get('prefix', '')}{sub}" for sub in all_subjects)]
        logging.warning(f"not all subjects' data could be found, missing subjects: {missing_subjects}")
    else:
        logging.debug(f"found all {len(all_subjects)} subjects")

    test_subs = settings.get("test_n_subs", 0)
    if test_subs > 0:
        all_subjects = np.random.choice(all_subjects, test_subs, replace=False)
        logging.debug(f"running on subjects: {all_subjects}")

    out_list = load_subjects(all_subjects, config, settings)
    list_map = {subject: i for i, subject in enumerate(all_subjects)}
    return list_map, out_list

