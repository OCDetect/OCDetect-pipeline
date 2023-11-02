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


def evaluate_single_model(model, param_grid,
                          X_train, y_train, X_test, y_test, feature_names,
                          cv_splits=8, cv_scoring=None, select_features=False,
                          out_dir='results/default', sample_balancing=None, seed=42, test_subject=None):

    os.makedirs(f'{out_dir}/test_subject_{test_subject}/val/', exist_ok=True)
    os.makedirs(f'{out_dir}/test_subject_{test_subject}/test/', exist_ok=True)
    model_name = str(model.__class__.__name__)

    cv = LeaveOneGroupOut()

    # Define list with steps for the pipeline
    pipeline_steps = []

    # ================= ADD BALANCING TO PIPELINE IF SELECTED =================
    if sample_balancing in ['random_undersampling', 'SMOTE']:
        logger.info(f'n samples before: {len(y_train[y_train == 0])} vs. {len(y_train[y_train == 1])}')
        if sample_balancing == 'random_undersampling':
            resampler = RandomUnderSampler()
            logger.info("Using random undersampling")
        else:  # 'SMOTE'
            resampler = SMOTE(n_jobs=-1, sampling_strategy=0.2689, random_state=seed)
            logger.info("Using oversampling")
        pipeline_steps.append(('resampling', resampler))

    # ================= SELECT OPTIMAL MODEL AND FEATURE SET THROUGH CV =================

    # prepare param_grid
    param_grid = {'model__' + key: value for (key, value) in param_grid.items()}

    if select_features:
        param_grid['selector'] = [SelectKBest(k='all'), SelectKBest(k=25),
                                  SelectFromModel(LinearSVC(C=1, penalty="l1", dual=False, max_iter=5000))]
        pipeline_steps.extend([('selector', 'passthrough'), ('model', model)])
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
        with open(f'{out_dir}/test_subject_{test_subject}/best_parameters.txt', 'a+') as f:
            f.write('\n' + model_name)
            f.write(f'GridSearch Failed due to incompatible options in best selected model.\n')
        logger.error("Warning: 'GridSearch Failed due to incompatible options in best selected model.")
        empty_cm = np.zeros((2, 2))
        return {metric: ([0.0] if metric != 'confusion_matrix' else [empty_cm] * cv_splits) for metric in all_metrics_list}, \
               {metric: (0.0 if metric != 'confusion_matrix' else empty_cm) for metric in all_metrics_list}, \
               (([0] * 101, [0] * 101, [0] * 101), ([0] * 101, [0] * 101, [0] * 101))
    with open(f'{out_dir}/test_subject_{test_subject}/best_parameters.txt', 'a+') as f:
        f.write('\n' + model_name)
        f.write(f'\nBest Params: {grid_model.best_params_}\n')
    logger.info(f'Best Params: {grid_model.best_params_} - {cv_scoring}: {grid_model.best_score_}')
    best_model = grid_model.best_estimator_


    # =================== Final Model Testing ===============
    X_test = X_test.drop(columns=["user"])
    test_metrics, test_curves = test_classification_model(best_model, X_train, y_train, X_test, y_test, feature_names, test_subject,
                                                          model_name, select_features, out_dir)
    return test_metrics, test_curves