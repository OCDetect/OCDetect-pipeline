from sklearn.metrics import *


all_classification_metrics_list = ['balanced_accuracy', 'recall', 'precision', 'mcc', 'f1_score', 'roc_auc', 'prc_auc',
                    'avg_precision', 'confusion_matrix']


def compute_classification_metrics(y_true, y_pred, binary_classification):
    # roc_auc_score and average_precision_score require probabilities instead of predictions
    if binary_classification:
        results = {'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
               'recall': recall_score(y_true, y_pred),
               'precision': precision_score(y_true, y_pred, zero_division=0),
               'mcc': matthews_corrcoef(y_true, y_pred),
               'f1_score': f1_score(y_true, y_pred),
               'confusion_matrix': confusion_matrix(y_true, y_pred)
               }
    else:
        results = {'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
                   'recall': recall_score(y_true, y_pred, average="macro"),
                   'precision': precision_score(y_true, y_pred, average="macro", zero_division=0),
                   'mcc': matthews_corrcoef(y_true, y_pred),
                   'f1_score': f1_score(y_true, y_pred, average="macro"),
                   'confusion_matrix': confusion_matrix(y_true, y_pred)
                   }
    return results
