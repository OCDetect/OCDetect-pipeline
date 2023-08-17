import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay
import numpy as np

# def plot_roc_pr_curve(X_test, y_test, endpoint, model, model_name, out_dir):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#     ax1.set_aspect('equal')
#     ax2.set_aspect('equal')
#     ax1.set(xlim=[-0.05, 1.05], ylim=[0.0, 1.05])
#     ax2.set(xlim=[-0.05, 1.05], ylim=[0.0, 1.05])
#     fig.suptitle(f'{model_name} predicting {endpoint}')
#     # ROC
#     # TODO plot_roc_curve is outdated (use e.g. RocCurveDisplay.from_predictions)
#     roc_plot = plot_roc_curve(model, X_test, y_test,
#                               name='ROC curve', lw=1, ax=ax1)
#     ax2.set(xlim=[-0.05, 1.05], ylim=[0.0, 1.05])
#     prc_plot = PrecisionRecallDisplay.from_estimator(model, X_test, y_test,
#                                                      name='PR curve', lw=1, ax=ax2)
#     plt.savefig(f'{out_dir}/{endpoint}/test/{model_name}_roc_prc_curves'.replace(' ', '_'), bbox_inches='tight', dpi=96)
#
#     return roc_plot, prc_plot


def plot_confusion_matrix(label, confusion_matrix, model_name, out_dir, phase):
    cm_fig, ax = plt.subplots()
    cm_fig.suptitle(f'{model_name} predicting {label}')
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                  display_labels=[0, 1])
    disp.plot(include_values=True, cmap='Blues', ax=ax,
              xticks_rotation='horizontal', values_format='d')
    plt.savefig(f'{out_dir}/{label}/{phase}/{model_name}_cm'.replace(' ', '_'))
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
    plt.savefig(f'{out_dir}/{label_name.replace(" ", "_")}/test/{model_name}_feature_importance', dpi=300)
    plt.close()
