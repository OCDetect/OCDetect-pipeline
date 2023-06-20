import os.path

import pandas as pd
from typing import List, Dict
from tqdm import tqdm


def export_data(dfs: List[pd.DataFrame], config: Dict, settings: Dict, subject: str, filenames : List[str]) -> None:
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
    :param filenames: List of recording filenames, in the same order as the recordings in dfs.
    :return: nothing
    """

    export_subfolder = config.get("export_subfolder")
    if not(os.path.isdir(export_subfolder)):
        os.mkdir(export_subfolder)
    # TODO: dfs = sorted(dfs, key= lambda x: recording_datetime(x), ascending=True)  # sort by date and
    # TODO: load metadata table and add recording, if it isnt already in there.

    for i, recording in tqdm(enumerate(dfs)):
        rec_id = str(i)
        while len(rec_id) < 2:
            rec_id = "0" + rec_id

        recording.to_csv(f"{export_subfolder}OCDetect_{subject}_recording_{rec_id}.csv", index=False)

