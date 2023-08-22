import os
import re
import time
import yaml
import pandas as pd
from typing import List
from tqdm import tqdm
from misc import logger
from tsfresh.feature_extraction import extract_features, MinimalFCParameters
from datetime import date
from data_preparation.utils.filter import butter_filter
from data_preparation.utils.scaler import std_scaling_data

save_data = True
overwrite_data = True


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
                majority_label = perform_majority_voting(curr_window)
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


def perform_majority_voting(current_window, hw_general=True):
    counts = current_window['relabeled'].value_counts()

    if hw_general:
        null_class = counts.get(0, 0)
        routine_hw = counts.get(1, 0)
        compulsive_hw = counts.get(2, 0)

        if routine_hw + compulsive_hw > null_class:
            majority_label = 1
        else:
            majority_label = 0
    else:
        null_class = counts.get(0, 0)
        routine_hw = counts.get(1, 0)
        compulsive_hw = counts.get(2, 0)

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


def load_data_preparation_settings(settings: dict): # Todo: make sure that not over- AND undersampling are True
    use_filter = settings.get("use_filter")
    use_scaling = settings.get("use_scaling")
    resample = settings.get("resample")
    use_undersampling = settings.get("use_undersampling")
    use_oversampling = settings.get("use_oversampling")

    return use_filter, use_scaling, resample, use_undersampling, use_oversampling


def get_data_path_variables(use_scaling, use_filter, config:dict, settings: dict):

    export_path = config.get("export_subfolder_ml_prepared")

    window_size = settings.get("window_size")
    subjects = settings.get("use_ocd_only")

    subjects_folder_name = "all_subjects" if not subjects else "ocd_diagnosed_only"
    sub_folder_path = f"ws_{window_size}_s/{subjects_folder_name}"

    scaling = "scaled" if use_scaling else "not_scaled"
    filtering = "filtered" if use_filter else "not_filtered"

    return window_size, subjects, subjects_folder_name, sub_folder_path, export_path, scaling, filtering


# main function for data preparation
def prepare_data(settings: dict, config: dict):
    use_filter, use_scaling, resample, use_undersampling, use_oversampling = load_data_preparation_settings(settings)
    all_subjects = True if not settings.get("use_ocd_only") else False

    logger.info("Preparing data for machine learning")

    folder_path = config.get("export_subfolder")
    pattern = r'OCDetect_(\d+)'

    dataframes = {}

    subject_numbers = settings.get("all_subjects") if all_subjects else settings.get("ocd_diagnosed_subjects")

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            match = re.search(pattern, file_name)
            if match:
                subject_number = int(match.group(1))

                if subject_number in subject_numbers:
                    file_path = os.path.join(folder_path, file_name)
                    df = pd.read_csv(file_path, )

                    if subject_number in dataframes:
                        dataframes[subject_number].append(df)
                    else:
                        dataframes[subject_number] = [df]

    subjects = dataframes.keys()
    data = list(dataframes.values())

    start_time = time.time()

    # 1. Filter data if desired
    if use_filter:
        filtered_data_all = []
        for subject in data:
            filtered_data_subject = []
            for recording in subject:
                filtered_data_subject.append(butter_filter(recording, settings))
            filtered_data_all.append(filtered_data_subject)
        data = filtered_data_all

    features = []
    labels = []
    users = []

    logger.info("Windowing data")
    for i, subject_data in zip(subjects, data):
        logger.info(f"Subject: {i} ----")

        # 2. Do some preparations for good measure
        logger.info("Sorting data")
        subject_data = sorted(subject_data, key=lambda x: pd.to_datetime(x.iloc[0].timestamp))

        # 3. Window data
        windows, user_labels, user_id = window_data(subject_data, i, settings)
        labels.append(user_labels)
        users.append(user_id)

        logger.info(f"Amount of data points : {len(windows)}")
        logger.info(f"Amount of windows : {windows['tsfresh_id'].iloc[-1]}")

        # 4. Extracting features
        features_user = feature_extraction(windows, settings)
        features.append(features_user)

        logger.info(f"Subject: {i}, features: {len(features_user)}, labels: {len(user_labels)}")

    labels = pd.concat(labels).reset_index(drop=True)
    users = pd.concat(users).reset_index(drop=True)

    features = pd.concat(features)
    feature_names = features.columns.values.tolist()

    # 5. Scale data if desired
    if use_scaling:
        features = std_scaling_data(features, settings)

    end_time = time.time()
    windowing_time_s = end_time - start_time
    windowing_time_min = windowing_time_s / 60

    if save_data:
        window_size, subjects, subjects_folder_name, sub_folder_path, export_path, scaling, filtering = get_data_path_variables(
            use_scaling, use_filter, config, settings)

        today = date.today()
        file_date = today.strftime("%Y-%m-%d")  # Format the date as "YYYY-MM-DD"

        os.makedirs(f"{export_path}/{sub_folder_path}", exist_ok=True)

        labels.to_csv(f"{export_path}{sub_folder_path}/labels_{filtering}_{scaling}.csv")
        users.to_csv(f"{export_path}{sub_folder_path}/users_{filtering}_{scaling}.csv")
        features.to_csv(f"{export_path}{sub_folder_path}/features_{filtering}_{scaling}.csv")
        pd.DataFrame(feature_names).to_csv(f"{export_path}{sub_folder_path}/feature_names_{filtering}_{scaling}.csv")

        # create file with meta information for the current window setup
        meta_info = f"Meta information for \"{file_date}\":\n"
        meta_info += "=" * 40 + "\n"
        meta_info += f"Time needed for window data: {windowing_time_s:.2f} seconds ({windowing_time_min:.2f} minutes) \n"
        meta_info += "-" * 40 + "\n"

        settings_data = yaml.dump(settings)

        meta_file = "meta_info.txt"
        with open(f"{export_path}/{sub_folder_path}/{meta_file}", 'w') as file:
            file.write(meta_info)
            file.write("Used following settings file: \n")
            file.write(settings_data)

    return labels, users, features, feature_names