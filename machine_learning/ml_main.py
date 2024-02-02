from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from misc import logger
import pandas as pd
import numpy as np
from collections import Counter
from machine_learning.classify.models import get_classification_model_grid
from machine_learning.classify.evaluate import evaluate_single_model
from .dl.dl_main import dl_main
from .dl.OCDetectDataset import OCDetectDataset
from sklearn.model_selection import StratifiedKFold as SKF


def ml_pipeline(features, users, labels, feature_names, seed, settings: dict, config: dict):
    # todo test this for case that data is not prepared
    try:
        users.rename(columns={'0': 'user'}, inplace=True)
    except:
        pass
    if len(feature_names) == 7:
        users_rep = users.loc[users.index.repeat(int(settings["window_size"] * 50))].to_numpy()
        features["user"] = users_rep
        windows = []
        for ind, window_df in features.groupby(["user", "tsfresh_id"]):
            windows.append(window_df.iloc[:,:6].to_numpy())
        windows = np.stack(windows)
    else:
        X = pd.merge(features, users, left_index=True, right_index=True)
        X.columns = X.columns.astype(str)
        labels = labels.iloc[:, 0]

    all_subjects = True if not settings.get("use_ocd_only") else False
    split_type = "all"
    if settings.get("use_ocd_only", False):
        split_type = "ocd_only"
    if settings.get("use_trustworthy_only", True):
        split_type = "trustworthy_only"
    balancing_option = settings.get("balancing_option")

    out_dir = config.get("ml_results_folder")
    only_dl = True ## TODO: set to false in settings - could also use "Raw" param.
    if only_dl:
        OCDetectDataset.preload(windows, users, labels)
        dl_main(config, settings, users, split_type)
        return


    # TODO add models to settings and read out which model(s) should be used if not all
    users = users["user"]
    users_outer_cv = list(users.unique())
    for test_subject in users_outer_cv:
        X_test = X[users == test_subject]
        y_test = labels[users == test_subject]
        X_train = X[users != test_subject]
        y_train = labels[users != test_subject]

        all_model_metrics = {}

        # model grid
        model_grid = get_classification_model_grid('balanced' if balancing_option == 'class_weight' else None, seed=seed)
        for j, (model, param_grid) in enumerate(model_grid):
            # val_metrics, test_metrics, curves = evaluate_single_model(model, param_grid,
            test_metrics, curves = evaluate_single_model(model, param_grid,
                                                                      X_train, y_train, X_test, y_test,
                                                                      out_dir=out_dir,
                                                                      sample_balancing=balancing_option,
                                                                      seed=seed)
            # all_model_metrics[str(model.__class__.__name__)] = (val_metrics, test_metrics, curves)
            all_model_metrics[str(model.__class__.__name__)] = (test_metrics, curves)