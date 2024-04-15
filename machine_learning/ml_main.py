import json

import pandas as pd
import numpy as np
from collections import Counter
from machine_learning.classify.models import get_classification_model_grid
from machine_learning.classify.evaluate import evaluate_single_model
from machine_learning.dl.dl_main import dl_main
from machine_learning.dl.OCDetectDataset import OCDetectDataset

import yaml
import shutil
from machine_learning.utils.plots import boxplot, barchart


def ml_pipeline(features, users, labels, feature_names, seed,settings: dict, config: dict, classic: bool = True):
    try:
        users.rename(columns={'0': 'user'}, inplace=True)
    except:
        pass
    if not classic:
        users_rep = users.loc[users.index.repeat(int(settings["window_size"] * 50))].to_numpy()
        features["user"] = users_rep
        windows = []
        for ind, window_df in features.groupby(["user", "tsfresh_id"]):
            windows.append(window_df.iloc[:, :6].to_numpy())
        windows = np.stack(windows)
    else:
        features.columns = feature_names
        users.columns = ["user"]
        X = pd.merge(features, users, left_index=True, right_index=True)
        labels = labels.iloc[:, 0]

    subject_groups_folder_name = settings.get("selected_subject_option")
    ws_folder_name = f"ws_{settings.get('window_size')}"
    # output folder in form like, e.g.: ml_results/all_subjects/ws_10/
    out_dir = f"{config.get('ml_results_folder')}/{subject_groups_folder_name}/{ws_folder_name}"

    balancing_option = settings.get("balancing_option")

    only_dl = settings.get("raw_features")
    if only_dl:
        OCDetectDataset.preload(windows, users, labels)
        dl_main(config, settings, users, subject_groups_folder_name)
        return

    balancing_option = settings.get("balancing_option")

    users = users["user"]
    users_outer_cv = list(users.unique())

    subject_metrics = {} # new dictionary to store all_model_metrics for specific test subject

    for test_subject in users_outer_cv:
        X_test = X[users == test_subject]
        y_test = labels[users == test_subject]
        X_train = X[users != test_subject]
        y_train = labels[users != test_subject]

        print("left out subject", test_subject)
        all_model_metrics = {}

        # model grid
        selected_models = [list(classifier.keys())[0] for classifier in settings.get("models") if list(classifier.values())[0]]
        model_grid = get_classification_model_grid(settings.get("all_models"), selected_models, seed=seed)
        for j, (model, param_grid) in enumerate(model_grid):
            test_metrics, curves = evaluate_single_model(model, param_grid,
                                                         X_train, y_train, X_test, y_test, feature_names,
                                                         out_dir=out_dir,
                                                         resample=resample,
                                                         sample_balancing=balancing_option,
                                                         seed=seed, test_subject=test_subject)
            all_model_metrics[str(model.__class__.__name__)] = (test_metrics, curves)
        subject_metrics[test_subject] = all_model_metrics

    barchart(out_dir, subject_metrics, settings.get("barchart_metric"))

    export_path = config.get("export_subfolder_ml_prepared")
    window_size = settings.get("window_size")

    source_file = f"{export_path}/ws_{window_size}_s/{subject_groups_folder_name}/meta_info.txt"
    destination_file = f"{out_dir}/meta_info.txt"

    try:
        shutil.copy(source_file, destination_file)
    except FileNotFoundError:
        print("Meta settings file not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    # ===== Save aggregate plots across models =====
    # Generate Boxplots for Metrics
    json_metric_data = {}
    for metric_name in all_model_metrics[str(model.__class__.__name__)][0].keys():
        if metric_name == 'confusion_matrix':
            json_metric_data[metric_name] = {model_name: (test_metrics[metric_name].tolist())
                                             for model_name, (test_metrics, _) in all_model_metrics.items()}
            continue
        metric_data = {model_name: (test_metrics[metric_name])
                       for model_name, (test_metrics, _) in all_model_metrics.items()}
        json_metric_data[metric_name] = metric_data
        # boxplot(out_dir, metric_data, metric_name, "hand washing", ymin=(-1 if metric_name == 'mcc' else 0))
    json.dump(json_metric_data, open(f'{out_dir}/all_model_metrics.json', 'w'), indent=4)
