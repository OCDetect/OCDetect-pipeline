from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from misc import logger
import pandas as pd
from collections import Counter
from machine_learning.classify.random_forest import random_forest_classifier


def ml_pipeline(features, users, labels, feature_names, all_subjects, resample, use_oversampling, use_undersampling, config: dict):
    if resample:
        # todo test this for case that data is not prepared
        users.rename(columns={'0': 'user'}, inplace=True)
        X = pd.merge(features, users, left_index=True, right_index=True)
        X.columns = X.columns.astype(str)

        labels = labels.iloc[:, 0]

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

    best_model, best_param, best_score = random_forest_classifier(X_res, y_res, users, config.get("plots_folder"), config.get("ml_results_folder"), all_subjects=all_subjects)
