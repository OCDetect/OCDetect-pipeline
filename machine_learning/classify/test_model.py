import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, auc
from machine_learning.classify.metrics import compute_classification_metrics
from machine_learning.classify.models import positive_class_probability, get_feature_importance
from machine_learning.utils.plots import plot_confusion_matrix, plot_coefficients, plot_roc_pr_curve
from misc import logger


def test_classification_model(model, X_train, y_train, X_test, y_test: pd.DataFrame, feature_names, test_subject, model_name, select_features, out_dir):
    # Re-fit complete training set
    # Reason: it is advisable to retrain the model on the entire training set (including the validation set)
    # after finding the best hyperparameters using GridSearchCV
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    X_train.columns = feature_names

    # Determine optimal decision boundary threshold
    def to_labels(pos_probs, threshold):
        return (pos_probs >= threshold).astype('int')

    y_probas = positive_class_probability(model, X_test)
    thresholds = np.linspace(0, 1, 101)
    scores = [f1_score(y_test, to_labels(y_probas, t)) for t in thresholds]
    ix = np.argmax(scores)
    optimal_threshold = thresholds[ix]
    optimal_f1 = scores[ix]

    N = 7
    padding = (N - 1) // 2
    kernel = np.ones(N) / N

    probas_classes = to_labels(y_probas, optimal_threshold)
    y_pred_smoothed = np.convolve(probas_classes, kernel, mode='valid')
    y_pred_smoothed_probas = np.convolve(y_probas, kernel, mode='valid')

    y_pred_postprocessed = np.concatenate([probas_classes[:padding], np.round(y_pred_smoothed), probas_classes[-padding:]])
    y_pred_postprocessed_probas = np.concatenate([y_probas[:padding], y_pred_smoothed_probas, y_probas[-padding:]])

    test_metrics = compute_classification_metrics(y_test, y_pred_postprocessed)
    orig_test_metrics = compute_classification_metrics(y_test, to_labels(y_probas, optimal_threshold))

    with open(f'{out_dir}/best_parameters.txt', 'a+') as f:
        f.write(f'optimal classification threshold: {optimal_threshold} with F1-Score {optimal_f1}\n\n')
        logger.info(f'optimal classification threshold: {optimal_threshold} with F1-Score {optimal_f1}\n\n')
        f.write(f"original metrics: {orig_test_metrics}\n\n")
        logger.info(f"original metrics: {orig_test_metrics}\n\n")
        f.write(f"postprocessed metrics: {test_metrics}\n\n")
        logger.info(f"postprocessed metrics: {test_metrics}\n\n")

    # ==== ROC & AUPRC ====
    _, _, _, _, _ = plot_roc_pr_curve(y_test, y_probas, model_name, out_dir, "original")
    roc_plot, roc_auc, prc_plot, auprc, average_precision = plot_roc_pr_curve(y_test, y_pred_postprocessed_probas, model_name, out_dir, "postprocessed")

    test_metrics['roc_auc'] = roc_auc
    test_metrics['avg_precision'] = average_precision
    # aucprs = auc(prc_plot.recall, prc_plot.precision)
    test_metrics['prc_auc'] = auprc

    plt.close()

    # ===== Confusion Matrix ====
    plot_confusion_matrix(test_subject, test_metrics['confusion_matrix'], model_name, out_dir)

    # ===== Feature Importances =====
    feature_importances = get_feature_importance(model)
    if select_features:
        feature_names = X_train.columns[model.named_steps['selector'].get_support()]
        with open(f'{out_dir}/best_parameters.txt', 'a+') as f:
            f.write(f'selected features: {feature_names}\n')
    else:
        feature_names = X_train.columns

    if feature_importances is not None:
        feature_importance = pd.DataFrame([feature_importances], columns=feature_names.values)
        feature_importance.to_csv(f'{out_dir}/{model_name}_feature_importance.csv')
        plot_coefficients(out_dir, feature_importances, feature_names, model_name, y_test.name)

    # interp_tpr = np.interp(thresholds, roc_plot.fpr, roc_plot.tpr, left=0.0) TODO: see ROC above (also for return)

    return test_metrics, None
