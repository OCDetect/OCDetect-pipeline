import os
import re
import time
import yaml
import pandas as pd
from typing import List
from tqdm import tqdm
from misc import logger
from tsfresh.feature_extraction import extract_features, MinimalFCParameters, ComprehensiveFCParameters
from datetime import date
from data_preparation.utils.filter import butter_filter
from data_preparation.utils.scaler import std_scaling_data
from tsfresh.utilities.dataframe_functions import impute


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


def perform_majority_voting(current_window, hw_general=True, hw_type=False):
    #counts = current_window['relabeled'].value_counts()
    # ToDo select column based on relabel mechanism
    counts = current_window['merged_annotation'].value_counts()

    # Count occurrences of N/A and 0, and 1, 2, 3, 4
    count_result = current_window['merged_annotation'].value_counts().reset_index(name='count')

    grouped_counts = []
    if hw_general:
        # Sum counts for N/A and 0, and 1, 2, 3, 4
        grouped_counts = count_result.groupby(lambda x: '0' if x in [float('nan'), 0] else '1').sum()
    if hw_type:
        grouped_counts = count_result.groupby(lambda x: '0' if x in [float('nan'), 0] else '1').sum()

    null_class = counts.get(0, 0)
    routine_hw = counts.get(1, 0)
    compulsive_hw = counts.get(2, 0)

    if hw_general:
        null = 0
        hw = 0
        if 0.0 in grouped_counts["index"].values:
            null = grouped_counts["count"][grouped_counts["index"] == 0.0].values[0]
        if 1.0 in grouped_counts["index"].values:
            hw = grouped_counts["count"][grouped_counts["index"] == 1.0].values[0]

        if hw >= null:
            majority_label = 1
        else:
            majority_label = 0

        # null_class = counts.get(0, 0)
        # routine_hw = counts.get(1, 0)
        # compulsive_hw = counts.get(2, 0)
        #
        # if routine_hw + compulsive_hw > null_class:
        #     majority_label = 1
        # else:
        #     majority_label = 0
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

    fc_parameters = {
        "mean": None,
        "standard_deviation": None,
        "maximum": None,
        "minimum": None,
        "abs_energy": None,
        "mean_abs_change": None,
        "absolute_sum_of_changes": None,
        "skewness": None,
        "kurtosis": None,
        "fft_aggregated": [{'aggtype': 'centroid'}, {'aggtype': 'variance'}, {'aggtype': 'skew'}, {'aggtype': 'kurtosis'}],
        "fourier_entropy": [{'bins': 2}, {'bins': 10}, {'bins': 100}]
    }
    # this did not create much better results but took long..
    fc_settings = {'variance_larger_than_standard_deviation': None,
                   'has_duplicate_max': None,
                   'has_duplicate_min': None,
                   'has_duplicate': None,
                   'sum_values': None,
                   'abs_energy': None,
                   'mean_abs_change': None,
                   'mean_change': None,
                   'mean_second_derivative_central': None,
                   'median': None,
                   'mean': None,
                   'length': None,
                   'standard_deviation': None,
                   'variation_coefficient': None,
                   'variance': None,
                   'skewness': None,
                   'kurtosis': None,
                   'root_mean_square': None,
                   'absolute_sum_of_changes': None,
                   'longest_strike_below_mean': None,
                   'longest_strike_above_mean': None,
                   'count_above_mean': None,
                   'count_below_mean': None,
                   'last_location_of_maximum': None,
                   'first_location_of_maximum': None,
                   'last_location_of_minimum': None,
                   'first_location_of_minimum': None,
                   'percentage_of_reoccurring_values_to_all_values': None,
                   'percentage_of_reoccurring_datapoints_to_all_datapoints': None,
                   'sum_of_reoccurring_values': None,
                   'sum_of_reoccurring_data_points': None,
                   'ratio_value_number_to_time_series_length': None,
                   'maximum': None,
                   'minimum': None,
                   'benford_correlation': None,
                   'time_reversal_asymmetry_statistic': [{'lag': 1}, {'lag': 2}, {'lag': 3}],
                   'c3': [{'lag': 1}, {'lag': 2}, {'lag': 3}],
                   'cid_ce': [{'normalize': True}, {'normalize': False}],
                   'symmetry_looking': [{'r': 0.0},
                                        {'r': 0.1},
                                        {'r': 0.2},
                                        {'r': 0.30000000000000004},
                                        {'r': 0.4},
                                        {'r': 0.5}],
                   'large_standard_deviation': [{'r': 0.5},
                                                {'r': 0.75},
                                                {'r': 0.9500000000000001}],
                   'quantile': [{'q': 0.1},
                                {'q': 0.2},
                                {'q': 0.3},
                                {'q': 0.4},
                                {'q': 0.6},
                                {'q': 0.7},
                                {'q': 0.8},
                                {'q': 0.9}],
                   'autocorrelation': [{'lag': 0},
                                       {'lag': 1},
                                       {'lag': 2},
                                       {'lag': 3},
                                       {'lag': 4},
                                       {'lag': 5},
                                       {'lag': 6},
                                       {'lag': 7},
                                       {'lag': 8},
                                       {'lag': 9}],
                   'agg_autocorrelation': [{'f_agg': 'mean', 'maxlag': 40},
                                           {'f_agg': 'median', 'maxlag': 40},
                                           {'f_agg': 'var', 'maxlag': 40}],
                   'partial_autocorrelation': [{'lag': 0},
                                               {'lag': 1},
                                               {'lag': 2},
                                               {'lag': 3},
                                               {'lag': 4},
                                               {'lag': 5},
                                               {'lag': 6},
                                               {'lag': 7},
                                               {'lag': 8},
                                               {'lag': 9}],
                   'number_cwt_peaks': [{'n': 1}, {'n': 5}],
                   'number_peaks': [{'n': 1}, {'n': 3}, {'n': 5}, {'n': 10}, {'n': 50}],
                   'binned_entropy': [{'max_bins': 10}],
                   'index_mass_quantile': [{'q': 0.1},
                                           {'q': 0.2},
                                           {'q': 0.3},
                                           {'q': 0.4},
                                           {'q': 0.6},
                                           {'q': 0.7},
                                           {'q': 0.8},
                                           {'q': 0.9}],
                   'spkt_welch_density': [{'coeff': 2}, {'coeff': 5}, {'coeff': 8}],
                   'ar_coefficient': [{'coeff': 0, 'k': 10},
                                      {'coeff': 1, 'k': 10},
                                      {'coeff': 2, 'k': 10},
                                      {'coeff': 3, 'k': 10},
                                      {'coeff': 4, 'k': 10},
                                      {'coeff': 5, 'k': 10},
                                      {'coeff': 6, 'k': 10},
                                      {'coeff': 7, 'k': 10},
                                      {'coeff': 8, 'k': 10},
                                      {'coeff': 9, 'k': 10},
                                      {'coeff': 10, 'k': 10}],
                   'value_count': [{'value': 0}, {'value': 1}, {'value': -1}],
                   'range_count': [{'min': -1, 'max': 1}],
                   'linear_trend': [{'attr': 'pvalue'},
                                    {'attr': 'rvalue'},
                                    {'attr': 'intercept'},
                                    {'attr': 'slope'},
                                    {'attr': 'stderr'}],
                   'augmented_dickey_fuller': [{'attr': 'teststat'},
                                               {'attr': 'pvalue'},
                                               {'attr': 'usedlag'}],
                   'number_crossing_m': [{'m': 0}, {'m': -1}, {'m': 1}],
                   'energy_ratio_by_chunks': [{'num_segments': 10, 'segment_focus': 0},
                                              {'num_segments': 10, 'segment_focus': 1},
                                              {'num_segments': 10, 'segment_focus': 2},
                                              {'num_segments': 10, 'segment_focus': 3},
                                              {'num_segments': 10, 'segment_focus': 4},
                                              {'num_segments': 10, 'segment_focus': 5},
                                              {'num_segments': 10, 'segment_focus': 6},
                                              {'num_segments': 10, 'segment_focus': 7},
                                              {'num_segments': 10, 'segment_focus': 8},
                                              {'num_segments': 10, 'segment_focus': 9}],
                   'ratio_beyond_r_sigma': [{'r': 0.5},
                                            {'r': 1},
                                            {'r': 1.5},
                                            {'r': 2},
                                            {'r': 2.5},
                                            {'r': 3},
                                            {'r': 5},
                                            {'r': 6},
                                            {'r': 7},
                                            {'r': 10}],
                   'linear_trend_timewise': [{'attr': 'pvalue'},
                                             {'attr': 'rvalue'},
                                             {'attr': 'intercept'},
                                             {'attr': 'slope'},
                                             {'attr': 'stderr'}],
                   'count_above': [{'t': 0}],
                   'count_below': [{'t': 0}],
                   'permutation_entropy': [{'tau': 1, 'dimension': 3},
                                           {'tau': 1, 'dimension': 4},
                                           {'tau': 1, 'dimension': 5},
                                           {'tau': 1, 'dimension': 6},
                                           {'tau': 1, 'dimension': 7}],
                   'query_similarity_count': [{'query': None, 'threshold': 0.0}]}

    logger.info("Extracting Features")
    features_list = extract_features(subject_windows, column_id=settings.get("id"),
                                              default_fc_parameters=fc_parameters,
                                              n_jobs=settings.get("jobs"))
    impute(features_list)  # make sure no NaNs are there
    return features_list


def load_data_preparation_settings(settings: dict):
    use_filter = settings.get("use_filter")
    use_scaling = settings.get("use_scaling")
    resample = settings.get("resample")
    balancing_option = settings.get("balancing_option")

    return use_filter, use_scaling, resample, balancing_option


def get_data_path_variables(use_scaling: object, use_filter: object, config: dict, settings: dict) -> object:

    export_path = config.get("export_subfolder_ml_prepared")

    window_size = settings.get("window_size")
    subjects_folder_name = settings.get("selected_subject_option")
    sub_folder_path = f"ws_{window_size}_s/{subjects_folder_name}"

    scaling = "scaled" if use_scaling else "not_scaled"
    filtering = "filtered" if use_filter else "not_filtered"

    return window_size, subjects_folder_name, sub_folder_path, export_path, scaling, filtering


# main function for data preparation
def prepare_data(settings: dict, config: dict, raw: str="both"):
    """
    :param settings: Global settings dict
    :param config:   Global user config dict
    :param raw:      "both": calculate and return features AND return raw windows
                     "raw" : only calculate and return raw windows
                     "features: only calculate and return features
    :return: The labels, features, users, feature names
    """
    save_data = settings["save_data"]
    overwrite_data = settings["overwrite_data"]
    use_filter, use_scaling, resample, _ = load_data_preparation_settings(settings)


    logger.info("Preparing data for machine learning")

    folder_path = config.get("export_subfolder") if settings['selected_subject_option'] != 'relabeled_subjects' else config.get("relabeled_subfolder")
    pattern = r'OCDetect_(\d+)'

    dataframes = {}

    selected_subject_option = str(settings['selected_subject_option'])
    subject_numbers = settings[selected_subject_option]

    for file_name in tqdm(os.listdir(folder_path)):
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
    features_raw = []
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
        if raw in ["both", "raw"]:
            features_raw.append(windows)
        if raw in ["both", "features"]:
            features_user = feature_extraction(windows, settings)
            features.append(features_user)

        length = max(len(features_user),len(windows))
        logger.info(f"Subject: {i}, features: {length}, labels: {len(user_labels)}")

    labels = pd.concat(labels).reset_index(drop=True).to_frame()
    users = pd.concat(users).reset_index(drop=True).to_frame()

    if raw in ["both", "features"]:
        features = pd.concat(features)

    if raw in ["both", "raw"]:
        features_raw = pd.concat(features_raw)

    try:
        feature_names = features.columns.values.tolist()
    except:
        feature_names = features_raw.columns.values.tolist()

    #print(str(len(features)))
    #features = pd.concat(features, ignore_index=True)
    #feature_names = features.columns.values.tolist()

    # 5. Scale data if desired (only on features)
    if use_scaling:
        features = std_scaling_data(features, settings)

    end_time = time.time()
    windowing_time_s = end_time - start_time
    windowing_time_min = windowing_time_s / 60

    if save_data:
        window_size, subjects_folder_name, sub_folder_path, export_path, scaling, filtering = get_data_path_variables(
            use_scaling, use_filter, config, settings)

        today = date.today()
        file_date = today.strftime("%Y-%m-%d")  # Format the date as "YYYY-MM-DD"

        os.makedirs(f"{export_path}/{sub_folder_path}", exist_ok=True)

        labels.to_csv(f"{export_path}{sub_folder_path}/labels_{filtering}_{scaling}.csv", index=False)
        users.to_csv(f"{export_path}{sub_folder_path}/users_{filtering}_{scaling}.csv", index=False)
        if len(features) > 0:
            features.to_csv(f"{export_path}{sub_folder_path}/features_{filtering}_{scaling}.csv", index=False)
        if len(features_raw) > 0:
            features_raw.to_csv(f"{export_path}{sub_folder_path}/features_{filtering}_raw.csv", index=False)
        pd.DataFrame(feature_names).to_csv(f"{export_path}{sub_folder_path}/feature_names_{filtering}_{scaling}.csv", index=False)

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

    return labels, [features, features_raw], users, feature_names