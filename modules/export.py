import os.path

import pandas as pd
from typing import List, Dict
from tqdm import tqdm

from helpers.logger import logger


def export_data(dfs: List[pd.DataFrame], config: Dict, settings: Dict, subject: str) -> None:
    """
    Function to export all data in the dfs-List to csv files.
    The export will be saved at the export path: config["export_subfolder"].
    1. Each participant's recordings will be stored in a sub-folder
    2. The recordings will be sorted by date and renamed that way.
    3. A metadata table will be exported to the root export path
    :param dfs: A list of DataFrames. Assumed to contain all relevant recordings for exactly one participant
    :param config: The global config dict
    :param settings: The global settings dict
    :param subject: The subject to export
    :return: nothing
    """

    export_subfolder = config.get("export_subfolder")
    if not(os.path.isdir(export_subfolder)):
        os.mkdir(export_subfolder)
    if os.path.isfile(export_subfolder + "exported.txt"):
        with open(export_subfolder + "exported.txt", "r") as f:
            for line in f:
                if line == subject:
                    return
    else:
        with open(export_subfolder + "exported.txt", "w") as f:
            pass

    dfs = sorted(dfs, key=lambda x: x.datetime.iloc[0])  # sort by date ascending

    column_whitelist = ["timestamp", "datetime", "acc x", "acc y", "acc z", "gyro x", "gyro y", "gyro z", "user yes/no",
                        "compulsive", "urge", "tense", "ignore", "relabeled"]

    meta_list = []
    for i, recording in tqdm(enumerate(dfs), total=len(dfs)):
        recording_id = recording.recording
        rec_number = str(i)

        while len(rec_number) < 2:
            rec_number = "0" + rec_number
        meta_list.append([subject, rec_number, recording_id, recording.datetime.iloc[0], len(recording) / 50])

        recording = recording[column_whitelist].copy()
        recording.relabeled = recording.relabeled.apply(lambda x: int(x))
        recording.ignore = recording.ignore.apply(lambda x: int(x))
        try:
            recording.to_csv(f"{export_subfolder}OCDetect_{subject}_recording_{rec_number}_{recording_id}.csv",
                             index=False)
        except Exception as e:
            logger.error(f"Error while exporting {recording_id}: {e}\ncontinuing...")
    new_meta_table = pd.DataFrame(meta_list, columns=["subject", "rec_no", "rec_id", "datetime", "duration"])
    meta_export_filename = export_subfolder + "recording_metadata_table.csv"
    if os.path.isfile(meta_export_filename):
        meta_table = pd.read_csv(meta_export_filename)
        meta_table = pd.concat([meta_table, new_meta_table]).reset_index(drop=True)
        meta_table = meta_table.drop_duplicates()  # makes sure running this again will not add the metadata again!
    else:
        meta_table = new_meta_table
    meta_table.to_csv(meta_export_filename, index=False)

    with open(export_subfolder + "exported.txt", "a") as f:
        f.write(subject + "\n")
