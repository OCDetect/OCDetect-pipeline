import os
import pandas as pd
from helpers.logger import logger
from imblearn.over_sampling import SMOTE
from machine_learning.classify import random_forest
from machine_learning.prepare.utils.scaler import std_scaling_data
from machine_learning.prepare.utils.filter import butter_filter

from machine_learning.prepare.data_preparation import butter_filter_data, window_data, feature_extraction


# todo: @ robin -> do you think we should put these conditions into a setup/settings file?
prepare_data = False
save_data = True

use_filter = True
use_scaling = True

use_oversampling = True


def do_ml(data, subjects, config: dict, settings: dict):
    logger.info("Starting machine learning pipeline")

    # if data is not prepared yet, do windowing etc. first
    if prepare_data:
        logger.info("Preparing data for machine learning")

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
        user_ids = []

        for i, subject_data in zip(subjects, data):
            logger.info(f"Subject: {i} ----")

            # 2. Do some preparations for good measure
            logger.info("Sorting data")
            subject_data = sorted(subject_data, key=lambda x: pd.to_datetime(x.iloc[0].timestamp))

            # 3. Window data
            windows, user_labels, user_id = window_data(subject_data, i, settings)
            labels.append(user_labels)
            user_ids.append(user_id)

            logger.info(f"Amount of data points : {len(windows)}")
            logger.info(f"Amount of windows : {windows['tsfresh_id'].iloc[-1]}")

            # 4. Extracting features
            features_user = feature_extraction(windows, settings)
            features.append(features_user)

            logger.info(f"Subject: {i}, features: {len(features_user)}, labels: {len(user_labels)}")

        labels = pd.concat(labels)
        user_ids = pd.concat(user_ids)
        features = pd.concat(features)
        feature_names = features.columns.values.tolist()

        # 5. Scale data if desired
        if use_scaling:
            features = std_scaling_data(features, settings)

        if save_data:
            export_path = config.get("export_subfolder_ml_prepared")
            # todo: create subfolder here with meta-data file (infos about window size, date and if scaling/filtering is enabled etc.)
            if not os.path.exists(export_path):
                os.makedirs(export_path)

            labels.to_csv(f"{export_path}labels.csv")
            user_ids.to_csv(f"{export_path}user_ids.csv")
            features.to_csv(f"{export_path}features.csv")
            pd.DataFrame(feature_names).to_csv(f"{export_path}feature_names.csv")

    else:
        # load prepared data
        logger.info("Read in prepared data")

        export_path = config.get("export_subfolder_ml_prepared")
        data = pd.read_csv(f"{export_path}features.csv").drop(columns=["Unnamed: 0"])
        labels = pd.read_csv(f"{export_path}labels.csv").drop(columns=["Unnamed: 0"])
        users = pd.read_csv(f"{export_path}user_ids.csv").drop(columns=["Unnamed: 0"])

    if use_oversampling:
        sm = SMOTE(random_state=42)
        data_w_users = data
        data_w_users['user'] = users

        X_res, y_res = sm.fit_resample(data_w_users, labels)
        users = X_res['user']
        X_res = X_res.drop(columns=["user"])

    else:
        X_res = data
        y_res = labels

    best_model, best_param, best_score = random_forest(X_res, y_res, users)

    # Use the best model for prediction or other tasks
    # y_pred = best_model.predict(X_test)

    # Create a confusion matrix
    # cm = confusion_matrix(y_true, y_pred)
