import os
import re
import time
import yaml
import pandas as pd
from helpers.logger import logger
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from machine_learning.prepare.utils.scaler import std_scaling_data
from machine_learning.prepare.utils.filter import butter_filter
from collections import Counter
from machine_learning.prepare.data_preparation import window_data, feature_extraction
from machine_learning.classify.random_forest import random_forest_classifier
from datetime import date

save_data = True
overwrite_data = True


def do_ml(config: dict, settings: dict, prepare_data, machine_learning):
    logger.info("Starting machine learning pipeline")

    use_filter, use_scaling, resample, use_undersampling, use_oversampling = load_ml_settings(settings)

    # if data is not prepared yet, do windowing etc. first
    if prepare_data:
        logger.info("Preparing data for machine learning")

        folder_path = config.get("export_subfolder")
        pattern = r'OCDetect_(\d+)'

        dataframes = {}

        all_subjects = True if not settings.get("use_ocd_only") else False
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
            window_size, subjects, subjects_folder_name, sub_folder_path, export_path, scaling, filtering = get_data_path_variables(use_scaling, use_filter, config, settings)

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

            meta_file = f"meta_info_{file_date}.txt"
            with open(f"{export_path}/{sub_folder_path}/{meta_file}", 'w') as file:
                file.write("Used settings file: \n")
                file.write(meta_info)
                file.write("Used following settings file: \n")
                file.write(settings_data)

    else:
        # load prepared data
        logger.info("Read in prepared data")

        window_size, subjects,  subjects_folder_name, sub_folder_path, export_path, scaling, filtering = get_data_path_variables(use_scaling, use_filter, config, settings)

        logger.info(f"Using path: {export_path}{sub_folder_path}")
        logger.info(f"Scaled data: {scaling}; Filtered data: {filtering}")

        print(f"{export_path}{sub_folder_path}/labels_{filtering}_{scaling}.csv")

        features = pd.read_csv(f"{export_path}{sub_folder_path}/labels_{filtering}_{scaling}.csv")
        labels = pd.read_csv(f"{export_path}{sub_folder_path}/users_{filtering}_{scaling}.csv")
        users = pd.read_csv(f"{export_path}{sub_folder_path}/features_{filtering}_{scaling}.csv")
        feature_names = pd.read_csv(f"{export_path}{sub_folder_path}/feature_names_{filtering}_{scaling}.csv")

    if machine_learning:
        if resample:
            X = features
            X['user'] = users
            X.columns = X.columns.astype(str)

            if use_oversampling:
                logger.info("Oversampling")
                logger.info(f"Before oversampling: {Counter(labels)}")
                sm = SMOTE(random_state=42)
                X_res, y_res = sm.fit_resample(X, labels)
                logger.info(f"After oversampling: {Counter(y_res)}")
            elif use_undersampling:
                logger.info("Undersampling")
                logger.info(f"Before undersampling: {Counter(labels)}")
                undersample = RandomUnderSampler(sampling_strategy='majority')
                X_res, y_res = undersample.fit_resample(X, labels)
                logger.info(f"After undersampling: {Counter(y_res)}")

            users = X_res['user']
            X_res = X_res.drop(columns=["user"])
        else:
            X_res = features
            y_res = labels

        best_model, best_param, best_score = random_forest_classifier(X_res, y_res, users)

    # Use the best model for prediction or other tasks
    # y_pred = best_model.predict(X_test)

    # Create a confusion matrix
    # cm = confusion_matrix(y_true, y_pred)


def load_ml_settings(settings: dict): # Todo: make sure that not over- AND undersampling are True
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

    return window_size, subjects,  subjects_folder_name, sub_folder_path, export_path, scaling, filtering
