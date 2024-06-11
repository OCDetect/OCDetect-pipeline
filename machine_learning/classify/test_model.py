import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, auc
from machine_learning.classify.metrics import compute_classification_metrics
from machine_learning.classify.models import positive_class_probability, get_feature_importance
from machine_learning.utils.plots import plot_confusion_matrix, plot_coefficients, plot_roc_pr_curve
from misc import logger
from sklearn.preprocessing import label_binarize
from collections import Counter


def test_classification_model(settings, model, X_train, y_train, X_test, y_test: pd.DataFrame, feature_names, test_subject, model_name, select_features, out_dir, binary_classification):
    multiclass_smoothing = settings.get("multiclass_pred_smoothing")
    window_size = settings.get("postprocessing_window_size")
    stride = settings.get("postprocessing_stride")

    # Re-fit complete training set
    # Reason: it is advisable to retrain the model on the entire training set (including the validation set)
    # after finding the best hyperparameters using GridSearchCV
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    X_train.columns = feature_names
    # Determine optimal decision boundary threshold
    def to_labels(pos_probs, threshold, binary_classification = True):
        if binary_classification:
            return (pos_probs >= threshold).astype('int')
        else:
            rout_probs = pos_probs[:, 1]
            comp_probs = pos_probs[:, 2]
            labels = np.zeros(len(pos_probs), dtype=int)
            labels[(rout_probs >= threshold) & (comp_probs < threshold)] = 1
            labels[(comp_probs >= threshold) & (rout_probs < threshold)] = 2
            labels[(rout_probs >= threshold) & (comp_probs >= threshold)] = np.argmax(
                pos_probs[(rout_probs >= threshold) & (comp_probs >= threshold)], axis=1)
            return labels

    y_probas = positive_class_probability(model, X_test, binary_classification)
    thresholds = np.linspace(0, 1, 101)
    if binary_classification:
        scores = [f1_score(y_test, to_labels(y_probas, t)) for t in thresholds]
    else:
        scores = [f1_score(y_test, to_labels(y_probas, t, binary_classification), average='macro') for t in thresholds]
    ix = np.argmax(scores)
    optimal_threshold = thresholds[ix]
    optimal_f1 = scores[ix]


    padding = (window_size - 1) // 2 if window_size % 2 == 1 else window_size // 2
    kernel = np.ones(window_size) / window_size

    probas_classes = to_labels(y_probas, optimal_threshold, binary_classification)

    if binary_classification:
        y_pred_smoothed = np.convolve(probas_classes, kernel, mode='valid')
        y_pred_postprocessed = np.concatenate([probas_classes[:padding], np.round(y_pred_smoothed), probas_classes[-padding:]])
        y_pred_smoothed_probas = np.convolve(y_probas, kernel, mode='valid')

    else:
        y_pred_smoothed_probas = np.zeros((y_probas.shape[0] - window_size + 1, y_probas.shape[1]))
        for i in range(y_probas.shape[1]):
            y_pred_smoothed_probas[:, i] = np.convolve(y_probas[:, i], kernel, mode='valid')
        y_pred_postprocessed = y_pred
        if multiclass_smoothing == "majority_class":
            for i in range(0, len(probas_classes) - window_size + 1, stride):
                window = probas_classes[i:i+window_size]
                counter = Counter(window)
                majority_value = counter.most_common(1)[0][0]
                y_pred_postprocessed[i:i+window_size] = [majority_value] * window_size
        elif multiclass_smoothing == "majority_proba":
            for i in range(0, len(y_probas) - window_size + 1, stride):
                window = y_probas[i:i + window_size]
                probas_sums = window.sum(axis=0)
                majority_proba_class = np.argmax(probas_sums)
                y_pred_postprocessed[i:i + window_size] = [majority_proba_class] * window_size

    y_pred_postprocessed_probas = np.concatenate([y_probas[:padding], y_pred_smoothed_probas, y_probas[-padding:]])

    test_metrics = compute_classification_metrics(y_test, y_pred_postprocessed, binary_classification)
    orig_test_metrics = compute_classification_metrics(y_test, to_labels(y_probas, optimal_threshold, binary_classification), binary_classification)


    with open(f'{out_dir}/best_parameters.txt', 'a+') as f:
        f.write(f'optimal classification threshold: {optimal_threshold} with F1-Score {optimal_f1}\n\n')
        logger.info(f'optimal classification threshold: {optimal_threshold} with F1-Score {optimal_f1}\n\n')
        f.write(f"original metrics: {orig_test_metrics}\n\n")
        logger.info(f"original metrics: {orig_test_metrics}\n\n")
        f.write(f"postprocessed metrics: {test_metrics}\n\n")
        logger.info(f"postprocessed metrics: {test_metrics}\n\n")

    # ==== ROC & AUPRC ====

    if binary_classification:
        _, _, _, _, _ = plot_roc_pr_curve(y_test, y_probas, model_name, out_dir, "original")
        roc_plot, roc_auc, prc_plot, auprc, average_precision = plot_roc_pr_curve(y_test, y_pred_postprocessed_probas, model_name, out_dir, "postprocessed")
    else:
        y_test_bin = label_binarize(y_test, classes = np.unique(y_test))
        roc_auc, average_precision, auprc = [], [], []
        for y_one_class in range(y_test_bin.shape[1]):
            _, _, _, _, _ = plot_roc_pr_curve(y_test_bin[:, y_one_class], y_probas[:, y_one_class], model_name, out_dir, f"original for one class {y_one_class}")
            roc_plot, roc_auc_class, prc_plot, auprc_class, average_precision_class = plot_roc_pr_curve(y_test_bin[:, y_one_class], y_pred_postprocessed_probas[:, y_one_class], model_name, out_dir, f"postprocessed for one class {y_one_class}")
            roc_auc.append(roc_auc_class)
            average_precision.append(average_precision_class)
            auprc.append(auprc_class)
        roc_auc, average_precision, auprc = np.mean(roc_auc), np.mean(average_precision), np.mean(auprc)

    test_metrics['roc_auc'] = roc_auc
    test_metrics['avg_precision'] = average_precision
    # aucprs = auc(prc_plot.recall, prc_plot.precision)
    test_metrics['prc_auc'] = auprc

    plt.close('all')

    # ===== Confusion Matrix ====
    plot_confusion_matrix(test_subject, test_metrics['confusion_matrix'], model_name, out_dir, binary_classification)

    # ===== Feature Importances =====
    feature_importances = get_feature_importance(model, binary_classification)
    if select_features:
        feature_names = X_train.columns[model.named_steps['selector'].get_support()]
        with open(f'{out_dir}/best_parameters.txt', 'a+') as f:
            f.write(f'selected features: {feature_names}\n')
    else:
        feature_names = X_train.columns

    if feature_importances is not None:
        model_name = str(model.__class__.__name__)
        if model_name == 'Pipeline':
            trained_model = model.named_steps['model']
            model_name = str(trained_model.__class__.__name__)
        if binary_classification or model_name not in ['LogisticRegression'] or (model_name == 'SVC' and model.named_steps['model'].kernel != 'linear'):
            feature_importance = pd.DataFrame([feature_importances], columns=feature_names.values)
            feature_importance.to_csv(f'{out_dir}/{model_name}_feature_importance.csv')
            plot_coefficients(out_dir, feature_importances, feature_names, model_name, y_test.name)
        else:
            feature_importance_df = pd.DataFrame(feature_importances, columns=feature_names.values)
            feature_importance_df.index = [f'{i}' for i in range(feature_importances.shape[0])]
            feature_importance_df.to_csv(f'{out_dir}/{model_name}_feature_importance.csv')
            for one_class in range(feature_importances.shape[0]):
                plot_coefficients(out_dir, feature_importance_df.iloc[one_class].values, feature_names, model_name, y_test.name, class_number=one_class)

    # interp_tpr = np.interp(thresholds, roc_plot.fpr, roc_plot.tpr, left=0.0) TODO: see ROC above (also for return)

    return test_metrics, None
