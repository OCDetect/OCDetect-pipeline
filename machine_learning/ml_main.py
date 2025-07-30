import json
import pandas as pd
import numpy as np
from machine_learning.classify.models import get_classification_model_grid
from machine_learning.classify.evaluate import evaluate_single_model
from machine_learning.dl.dl_main import dl_main
from machine_learning.dl.OCDetectDataset import OCDetectDataset
import shutil
from machine_learning.utils.plots import boxplot, barchart
from misc import logger

def ml_pipeline(features, users, labels, feature_names, seed, settings: dict, config: dict, classic: bool = True):
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

    label_type = settings.get("label_type")
    subject_groups_folder_name = settings.get("selected_subject_option")
    ws_folder_name = f"ws_{settings.get('window_size')}"
    # output folder in form like, e.g.: ml_results/all_subjects/ws_10/
    out_dir = f"{config.get('ml_results_folder')}/{subject_groups_folder_name}/{label_type}/{ws_folder_name}"


    if not classic:
        OCDetectDataset.preload(windows, users, labels)
        dl_main(config, settings, users, subject_groups_folder_name, out_dir)
        return

    balancing_option = settings.get("balancing_option")
    feature_selection = settings.get("feature_selection")

    users = users["user"]
    users_outer_cv = list(users.unique())

    subject_metrics = {} # new dictionary to store all_model_metrics for specific test subject

    for test_subject in users_outer_cv:
        X_test = X[users == test_subject]
        y_test = labels[users == test_subject]
        X_train = X[users != test_subject]
        y_train = labels[users != test_subject]

        logger.info(f"Current Test Subject: {test_subject}")
        all_model_metrics = {}

        # model grid
        selected_models = [list(classifier.keys())[0] for classifier in settings.get("models") if list(classifier.values())[0]]
        model_grid = get_classification_model_grid(settings.get("all_models"), selected_models, seed=seed)
        for j, (model, param_grid) in enumerate(model_grid):
            test_metrics, curves = evaluate_single_model(model, param_grid,
                                                         X_train, y_train, X_test, y_test, feature_names,
                                                         out_dir=out_dir,
                                                         select_features=feature_selection,
                                                         grid_search=settings.get("grid_search"),
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
    # json_metric_data = {}
    # for subject in subject_metrics:
    #     all_metrics_per_subject = subject_metrics[subject]
    #     for metric_name in all_metrics_per_subject[str(model.__class__.__name__)][0].keys():
    #         if metric_name == 'confusion_matrix':
    #             json_metric_data[metric_name] = {model_name: (test_metrics[metric_name].tolist())
    #                                              for model_name, (test_metrics, _) in all_metrics_per_subject.items()}
    #             continue
    #         metric_data = {model_name: (test_metrics[metric_name])
    #                        for model_name, (test_metrics, _) in all_metrics_per_subject.items()}
    #         json_metric_data[metric_name] = metric_data
    #         # boxplot(out_dir, metric_data, metric_name, "hand washing", ymin=(-1 if metric_name == 'mcc' else 0))
    # json.dump(json_metric_data, open(f'{out_dir}/all_model_metrics.json', 'w'), indent=4)

    # Initialize a dictionary to store average metric values
    average_metrics = {}

    # Iterate through each metric name
    for metric_name in all_model_metrics[str(model.__class__.__name__)][0].keys():
        if metric_name == 'confusion_matrix':
            # For confusion matrix, concatenate all matrices for each model across subjects
            confusion_matrices = {}
            for subject, metrics_per_subject in subject_metrics.items():
                for model_name, (test_metrics, _) in metrics_per_subject.items():
                    if model_name not in confusion_matrices:
                        confusion_matrices[model_name] = []
                    confusion_matrices[model_name].append(test_metrics[metric_name])

            # Calculate the average confusion matrix for each model
            average_confusion_matrices = {
                model_name: np.mean(confusion_matrices[model_name], axis=0).tolist()
                for model_name in confusion_matrices
            }

            # Store the average confusion matrices in the average_metrics dictionary
            average_metrics[metric_name] = average_confusion_matrices
        else:
            # For other metrics, calculate the average value across all subjects for each model
            metric_data = {}
            for subject, metrics_per_subject in subject_metrics.items():
                for model_name, (test_metrics, _) in metrics_per_subject.items():
                    if model_name not in metric_data:
                        metric_data[model_name] = []
                    print(metric_data.keys())
                    metric_data[model_name].append(test_metrics[metric_name])

            # Calculate the average metric value for each model
            average_metric_data = {
                model_name: np.mean(metric_data[model_name])
                for model_name in metric_data
            }

            # Store the average metric values in the average_metrics dictionary
            average_metrics[metric_name] = average_metric_data

    # Store the average metrics in a JSON file
    with open(f'{out_dir}/average_metrics.json', 'w') as f:
        json.dump(average_metrics, f, indent=4)