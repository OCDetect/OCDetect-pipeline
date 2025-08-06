import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def make_cm(y_test, y_pred, folder_path_plots, title, test_subject):
    cm = confusion_matrix(y_test, y_pred)
    class_names = ['NULL', 'HW']
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)

    plt.savefig(f'{folder_path_plots}_os_test_subject_{test_subject}.png')
    plt.close()

    return None
