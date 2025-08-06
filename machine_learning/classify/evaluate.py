import os
import time
from machine_learning.classify.metrics import all_classification_metrics_list, compute_classification_metrics
from machine_learning.classify.test_model import test_classification_model
from misc import logger
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from imblearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from collections import Counter



def evaluate_single_model(model, param_grid,
                          X_train, y_train, X_test, y_test, feature_names,
                          cv_splits=8, cv_scoring=None, select_features=False, grid_search=True,
                          out_dir='results/default', sample_balancing=None, seed=42, test_subject=None):


    subject_out_dir = f'{out_dir}/test_subject_{test_subject}/test/'
    os.makedirs(subject_out_dir, exist_ok=True)
    model_name = str(model.__class__.__name__)

    cv = LeaveOneGroupOut()

    # Define list with steps for the pipeline
    pipeline_steps = []

    class_ratio_one = len(y_train[y_train == 1]) / len(y_train)
    class_ratio_null = len(y_train[y_train == 0]) / len(y_train)
    if binary_classification:
        logger.info(f"The class ratio is: {class_ratio_one:.2f} vs. {class_ratio_null:.2f}")
    else:
        class_ratio_two = len(y_train[y_train == 2]) / len(y_train)
        logger.info(f"The class ratio is: {class_ratio_one:.2f} vs. {class_ratio_null:.2f} vs. {class_ratio_two:.2f} ")


    upsampling_ratio = 0.5
    downsampling_ratio = 0.5

    if not binary_classification:
        counts = Counter(y_train)
        if sample_balancing in ['SMOTE', 'SMOTETomek', 'SMOTEENN']:
            upsampling_ratio = {0: 1.0, 1: 0.5, 2: 0.5}
            logger.info(f"For upsampling the minority class, a class ratio of {upsampling_ratio} will be achieved.")
            max_count = max(counts.values())
            upsampling_ratio = {cls: int(max_count * ratio) for cls, ratio in upsampling_ratio.items()}
        elif sample_balancing == 'random_undersampling':
            downsampling_value = {0: 2.0, 1: 1.0, 2: 1.0}
            logger.info(f"For downsampling the majority class, a class ratio of {downsampling_value} will be achieved.")
            min_count = min(counts.values())
            downsampling_ratio = {cls: min(int(min_count * ratio), counts[cls]) for cls, ratio in downsampling_value.items()}


    if binary_classification and sample_balancing in ['SMOTE', 'SMOTETomek', 'SMOTEENN']:
        logger.info(f"For upsampling the minority class, a class ratio of {upsampling_ratio} will be achieved.")

    # ================= ADD BALANCING IF SELECTED =================
    resampler = None
    if sample_balancing in ['random_undersampling', 'SMOTE', 'SMOTETomek', 'SMOTEENN']:
        if binary_classification:
            logger.info(f'n samples before: {len(y_train[y_train == 0])} vs. {len(y_train[y_train == 1])}')
        else:
            logger.info(f'n samples before: {len(y_train[y_train == 0])} vs. {len(y_train[y_train == 1])} vs. {len(y_train[y_train == 2])}')
        if sample_balancing == 'random_undersampling':
            resampler = RandomUnderSampler(random_state=42, sampling_strategy=downsampling_ratio)
            logger.info("Using random undersampling")
        elif sample_balancing == 'SMOTE':  # sampling strategy: corresponds to the desired ratio of the number of samples in the minority class over the number of samples in the majority class after resampling
            resampler = SMOTE(n_jobs=-1, sampling_strategy=upsampling_ratio, random_state=seed)
            logger.info("Using oversampling")
        elif sample_balancing == 'SMOTETomek':
            resampler = SMOTETomek(sampling_strategy=upsampling_ratio, tomek=TomekLinks(sampling_strategy='majority'))
            logger.info("Using SMOTE and Tomek Links")
        elif sample_balancing == 'SMOTEENN':
            resampler = SMOTEENN(sampling_strategy=upsampling_ratio, enn=EditedNearestNeighbours(sampling_strategy='majority'))
            logger.info("Using SMOTE and edited nearest neighbours")
    else:
        logger.info("No additional balancing selected")

    if resampler is not None:
        X_train, y_train = resampler.fit_resample(X_train, y_train)

    # ================= SELECT OPTIMAL MODEL AND FEATURE SET THROUGH CV =================

    # prepare param_grid
    param_grid = {'model__' + key: value for (key, value) in param_grid.items()}

    logger.info(f"Amount of features before selection: {len(feature_names)}")

    if select_features and grid_search:
        param_grid['selector'] = [SelectKBest(k='all'), SelectKBest(k=10), SelectFromModel(RandomForestClassifier(random_state=42), threshold='median')]
        pipeline_steps.extend([('selector', 'passthrough'), ('model', model)])
    elif select_features and not grid_search:
        pipeline_steps.extend([
            ('selector', SelectFromModel(RandomForestClassifier(random_state=42), threshold='median')),
            ('model', model)
        ])
    else:
        pipeline_steps.append(('model', model))

    pipeline = Pipeline(pipeline_steps)
    logger.info(f"Using pipeline {pipeline}")

    # Define metrics used
    all_metrics_list = all_classification_metrics_list


    # Default CV scoring
    if cv_scoring is None:
        cv_scoring = "f1"

    # extract user column for cv and remove it for model training
    users = X_train['user']
    X_train = X_train.drop(columns=["user"])

    if grid_search:
        logger.info("Grid search cv running....")
        start_time = time.time()

        # n_jobs=-1 -> all CPUs are used to perform parallel tasks
        grid_model = GridSearchCV(pipeline, param_grid=param_grid, scoring=cv_scoring, cv=cv, n_jobs=-1)
        grid_model.fit(X_train, y_train, groups=users)

        end_time = time.time()
        logger.info("Grid search completed")

        training_time_seconds = end_time - start_time
        training_time_minutes = training_time_seconds / 60
        logger.info(f"Training time: {training_time_seconds:.2f} seconds ({training_time_minutes:.2f} minutes)")
        try:
            pass
        except ValueError as ve:
            logger.error(ve)
            with open(f'{subject_out_dir}/best_parameters.txt', 'a+') as f:
                f.write('\n' + model_name)
                f.write(f'GridSearch Failed due to incompatible options in best selected model.\n')
            logger.error("Warning: 'GridSearch Failed due to incompatible options in best selected model.")
            empty_cm = np.zeros((2, 2))
            return {metric: ([0.0] if metric != 'confusion_matrix' else [empty_cm] * cv_splits) for metric in all_metrics_list}, \
                   {metric: (0.0 if metric != 'confusion_matrix' else empty_cm) for metric in all_metrics_list}, \
                   (([0] * 101, [0] * 101, [0] * 101), ([0] * 101, [0] * 101, [0] * 101))
        with open(f'{subject_out_dir}/best_parameters.txt', 'a+') as f:
            f.write('\n' + model_name)
            f.write(f'\nBest Params: {grid_model.best_params_}\n')
        logger.info(f'Best Params: {grid_model.best_params_} - {cv_scoring}: {grid_model.best_score_}')
        best_model = grid_model.best_estimator_

    else:
        best_model = pipeline

    #  =================== Final Model Testing ===============
    X_test = X_test.drop(columns=["user"])
    test_metrics, test_curves = test_classification_model(settings, best_model, X_train, y_train, X_test, y_test, feature_names, test_subject,
                                                          model_name, select_features, subject_out_dir, binary_classification)
    return test_metrics, test_curves
