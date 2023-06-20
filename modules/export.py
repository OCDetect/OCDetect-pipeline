import os.path

import pandas as pd
import datetime
from glob import glob
from typing import List, Dict, Tuple
from pathlib import Path
from helpers.misc import get_metadata, add_timezone_and_summertime, get_file_name_initial_hw
from helpers.logger import logger
from helpers.definitions import IgnoreReason
from tqdm import tqdm
import numpy as np


def export_data(dfs: List[pd.DataFrame], config: Dict, settings: Dict) -> None:
    """
    Function to export all data in the dfs-List to csv files.
    The export will be saved at the export path: config["export_path"].
    1. Each participant's recordings will be stored in a sub-folder
    2. The recordings will be sorted by date and renamed that way.
    3. A metadata table will be exported to the root export path
    :param dfs: A list of DataFrames. Assumed to contain all relevant recordings for at least one participant
    :param config: The global config dict
    :param settings: The global settings dict
    :return: nothing
    """

    export_path = config.get("export_path")
    if not(os.path.isdir(export_path)):
        os.mkdir(export_path)



    pass