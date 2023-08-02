import pandas as pd
from enum import Enum
from typing import List
from data_cleansing.helpers.definitions import IgnoreReason, HandWashingType
from misc.csv_loader import initial_handwash_time


class RelabelStrategy(Enum):
    TimeSpan = 0
    InitialHandWashDuration = 1
    NearLabelSearch = 2


def relabel(dfs: List[pd.DataFrame], config: dict, settings: dict, subject: str, use_ignore: bool = True,
            strategy: RelabelStrategy = RelabelStrategy.TimeSpan) -> List[pd.DataFrame]:
    """
    Function to convert from 1 timestamp-after-hand-washing-labels to start and end, i.e. to region labels.
    :param dfs: The dataframes to which new labels should be applied
    :param config: The global config
    :param settings: The global settings
    :param subject: The subject on which we are running, relevant for the initial hand wash time.
    :param use_ignore: Whether to exclude labels for which the ignore column is set to a value other than DontIgnore
    :param strategy: How the new labels should be decided
    :return: The dataframe, with an additional column "relabeled"
    """
    relabeled_dfs = []
    if strategy == RelabelStrategy.NearLabelSearch:
        raise NotImplementedError(f"{strategy} is not implemented yet.")
    for df in dfs:
        hand_wash_df = df[df["user yes/no"] == 1]
        if use_ignore:
            hand_wash_df = hand_wash_df[hand_wash_df["ignore"] == IgnoreReason.DontIgnore]
        df["relabeled"] = HandWashingType.NoHandWash
        offset_samples = settings.get("relabel_offset", 5) * 50
        if strategy == RelabelStrategy.TimeSpan:
            duration_samples = settings.get("relabel_duration", 38) * 50

        if strategy == RelabelStrategy.InitialHandWashDuration:
            duration_samples = initial_handwash_time(subject, config) * 50
        for index, row in hand_wash_df.iterrows():
            start_index = index - duration_samples - offset_samples
            end_index = index - offset_samples
            df.loc[start_index:end_index, "relabeled"] = HandWashingType.Routine if row.compulsive == 0 \
                else HandWashingType.Compulsive
        relabeled_dfs.append(df)
    return relabeled_dfs
