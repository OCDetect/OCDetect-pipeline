import pandas as pd
from typing import List
from tqdm import tqdm
from misc import logger
from tsfresh.feature_extraction import extract_features, MinimalFCParameters


def window_data(subject_recordings: List[pd.DataFrame], subject_id, settings: dict):
    """
    Prepares the list with recordings for one subject by creating windows.
    :param subject_recordings: A list with single recordings for a subject
    :param subject_id: the ID for the subject
    :param settings: the overall settings
    :return: the data fragmented into windows
    """

    logger.info("Window Data")

    # get predefined settings
    window_size = settings.get("window_size") * settings.get("sampling_frequency")
    overlap = settings.get("overlap")
    labelling_algorithm = settings.get("labelling_algorithm")

    stride = int(window_size * (1 - overlap))

    window_labels = []
    window_list = []
    user_list = []

    unique_id = 0  # for calculating tsfresh features

    for recording in subject_recordings:
        user_data_size = len(recording)

        for w in tqdm(range(0, user_data_size - window_size + 1, stride)):
            # copy to avoid warning that original dataset is changed
            curr_window = recording.iloc[w:w + window_size].copy()
            curr_window = curr_window.fillna(0)

            # if current window only has entries that are labelled as to be ignored,
            # do not consider this window any further
            if check_ignore(curr_window):
                continue

            # Choose label
            if labelling_algorithm == 'Majority':
                majority_label = perform_majority_voting(settings.get("classification_mode"), curr_window)
                window_labels.append(majority_label)

                user_list.append(subject_id)
            curr_window['tsfresh_id'] = unique_id

            unique_id += 1
            curr_window = curr_window[['acc x', 'acc y', 'acc z', 'gyro x', 'gyro y', 'gyro z', 'tsfresh_id']]
            # pythons list append is much faster than pandas concat (time increased exponentially with concat)
            window_list.append(curr_window)

    return pd.concat(window_list), pd.Series(window_labels, index=None), pd.Series(user_list, index=None)


def check_ignore(current_window):
    counts = current_window['ignore'].value_counts()
    if counts.get(0, 0) > 0 or counts.get(1, 0) > 0:  # Todo: @ Robin, is that check sufficient?
        return False
    else:
        return True


def perform_majority_voting(classification_mode, current_window):
    counts = current_window['relabeled'].value_counts()

    null_class = counts.get(0, 0)
    routine_hw = counts.get(1, 0)
    compulsive_hw = counts.get(2, 0)

    if classification_mode == "binary":
        if routine_hw + compulsive_hw > null_class:
            majority_label = 1
        else:
            majority_label = 0
    elif classification_mode == "multiclass":
        if routine_hw > compulsive_hw and routine_hw > null_class:
            majority_label = 1  # routine hand washing present
        elif compulsive_hw > routine_hw and compulsive_hw > null_class:
            majority_label = 2  # compulsive hand washing present
        else:
            majority_label = 0  # null class

    return majority_label


def feature_extraction(subject_windows: pd.DataFrame, settings):
    """
    Extracts features using tsfresh
    :param subject_windows: The windowed data for one subject
    :param settings: the overall settings
    :return: the extracted features in a list
    """

    logger.info("Extracting Features")
    features_list = extract_features(subject_windows, column_id=settings.get("id"),
                                              default_fc_parameters=MinimalFCParameters(),
                                              n_jobs=settings.get("jobs"))

    return features_list
