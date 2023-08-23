import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, precision_recall_curve, roc_curve, auc, \
    average_precision_score
import numpy as np


def plot_roc_pr_curve(X_test, y_test, endpoint, model, model_name, out_dir):
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

    plt.savefig(f'{out_dir}/test/{model_name}_roc_prc_curves'.replace(' ', '_'), bbox_inches='tight', dpi=300)

    return roc_curve, roc_auc, precision_recall_curve, prc_auc, average_precision


def plot_confusion_matrix(label, confusion_matrix, model_name, out_dir, phase):
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
    plt.savefig(f'{out_dir}/test/{model_name}_feature_importance', dpi=300)
    plt.close()
