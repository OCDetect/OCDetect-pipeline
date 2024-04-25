import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, precision_recall_curve, roc_curve, auc, \
    average_precision_score
import numpy as np

model_name_replacements = {
    'LogisticRegression': 'Logistic regression',
    'GradientBoostingClassifier': 'Gradient boosting machine',
    'RandomForestClassifier': 'Random forest'
}

def plot_roc_pr_curve(X_test, y_test, model, model_name, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax1.set(xlim=[-0.05, 1.05], ylim=[0.0, 1.05])
    ax2.set(xlim=[-0.05, 1.05], ylim=[0.0, 1.05])
    fig.suptitle(f'{model_name}')

    # Calculate ROC curve and ROC AUC
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    # Calculate Average Precision
    average_precision = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])

    # Plot ROC curve
    ax1.plot(fpr, tpr, color='darkorange', lw=1, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic')
    ax1.legend(loc='lower right')

    # Calculate Precision-Recall curve and PR AUC
    precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
    prc_auc = auc(recall, precision)

    # Plot Precision-Recall curve
    ax2.plot(recall, precision, color='darkorange', lw=1, label=f'PR curve (AUC = {prc_auc:.2f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc='lower left')

    plt.savefig(f'{out_dir}/{model_name}_roc_prc_curves'.replace(' ', '_'), bbox_inches='tight', dpi=300)

    return roc_curve, roc_auc, precision_recall_curve, prc_auc, average_precision


def plot_confusion_matrix(test_subject, confusion_matrix, model_name, out_dir, phase):
    cm_fig, ax = plt.subplots()
    cm_fig.suptitle(f'{model_name}')
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                  display_labels=[0, 1])
    disp.plot(include_values=True, cmap='Blues', ax=ax,
              xticks_rotation='horizontal', values_format='d')
    plt.savefig(f'{out_dir}/{model_name}_cm'.replace(' ', '_'))
    plt.close()


def plot_coefficients(out_dir, coefs, feature_names, model_name, label_name, top_features=None):
    # filter out 0 values
    filter_mask = (abs(coefs) > 1e-3)
    coefs = coefs[filter_mask]
    feature_names = feature_names[filter_mask]
    if top_features:
        top_coef_indixes = np.argsort(abs(coefs))[-top_features:]
        coefs = coefs[top_coef_indixes]
        feature_names = feature_names[top_coef_indixes]
    # sort coefficients
    sort_mask = np.argsort(coefs)
    coefs = coefs[sort_mask]
    feature_names = feature_names[sort_mask]
    # plot
    plt.figure(figsize=(30, 10))
    plt.suptitle(f'Feature importance - {model_name} predicting {label_name}')
    colors = ['red' if c < 0 else 'blue' for c in coefs]
    plt.bar(np.arange(len(coefs)), coefs, color=colors)
    plt.xticks(np.arange(len(coefs)), feature_names, rotation=60, ha='right')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{model_name}_feature_importance', dpi=300)
    plt.close()


def boxplot(out_dir, data, metric_name, y_label, ymin=0, ymax=1):
    """Prints boxplot for CV Splits and additionally plots test set value

    Parameters
    ----------
    out_dir : str
        Base output directory
    data : dict
        Metric data in the form {model_name: (list of val split results, test split result)}
    metric_name : str
        The name of the metric
    y_label : str
        The name of the predicted endpoint
    ymin : int, default=0
        min of y axis (usually 0)
    ymax : int, default=0
        max of y axis (usually 1)
    """

    fig = plt.figure()
    fig.suptitle(f'{metric_name} for all models predicting {y_label}')
    ax = fig.add_subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    model_names = list(data.keys())
    # Plot val boxplot
    #val_data = list(map(lambda x: x[0], data.values()))
    val_data = list(data.values())
    plt.boxplot(val_data)

    # Plot test single data point
    #test_data = list(map(lambda x: x[1], data.values()))
    #plt.scatter(range(1, len(model_names) + 1), test_data, marker='o', color='blue')

    # Format axes etc
    ax.set_xticklabels([model_name_replacements.get(model_name, model_name) for model_name in model_names], rotation=45, ha='right')
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(metric_name)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/all_models_{metric_name}', dpi=96)
    plt.close()

def barchart(out_dir, data, metric_name):
    """Prints bar chart of metric_name (set in settings) for all test subjects and models

    Parameters
    out_dir : str
        Base output directory
    data : dict
        Metric data in the form {test_subject: (model: (metrics: value)}
    metric_name : str
        The name of the metric
    """

    subjects = []
    models = []
    metrics_value = []
    for subject_id, metrics in data.items():
        for model, value in metrics.items():
            subjects.append(subject_id)
            models.append(model)
            metrics_value.append(value[0][metric_name])

    # unique arrays for chart positioning
    unique_models = set(models)
    unique_subjects = set(subjects)

    # position and colors of the charts
    bar_positions = np.arange(len(unique_subjects)) * (len(unique_models) * 0.5)
    bar_width = 0.2
    colors = plt.cm.viridis_r(np.linspace(0, 1, len(unique_models)))

    fig, ax = plt.subplots(figsize=(12, 8))

    # matching metric_values and setting on the corresponding bars
    for i, model in enumerate(unique_models):
        model_data = []
        for j, m in enumerate(models):
            if m == model:
                model_data.append(metrics_value[j])
        ax.bar(bar_positions + i * bar_width, model_data, bar_width, label=model, color=colors[i], alpha=0.7)

    # set axes and legend
    ax.set_xticks(bar_positions + (len(unique_models) - 1) * bar_width / 2)
    ax.set_xticklabels(unique_subjects)
    ax.set_xlabel('test subjects')
    ax.set_ylabel(metric_name)
    ax.set_title(metric_name + ' per test subject and model')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    # save
    plt.savefig(f'{out_dir}/all_models_{metric_name}', dpi=96)
    plt.close()