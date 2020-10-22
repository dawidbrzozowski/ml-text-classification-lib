import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics


def _plot_multiple_precision_recall_curves(precision_recall_curves: dict):
    fig, axs = plt.subplots(len(precision_recall_curves))
    if len(precision_recall_curves) == 1:
        axs = [axs]
    fig.canvas.set_window_title('Precision Recall Curves')
    fig.suptitle('Precision Recall Curves')

    for i, model_name in enumerate(precision_recall_curves):
        precisions, recalls, thresholds = precision_recall_curves[model_name]
        axs[i].set_title(model_name)
        axs[i].plot(thresholds, precisions[:-1], 'b--', label='Precision')
        axs[i].plot(thresholds, recalls[:-1], 'g-', label='Recall')
        axs[i].set_xlabel('Threshold')
    plt.tight_layout()
    plt.ylim([0, 1])
    plt.legend(loc='center left')
    plt.show()


def _plot_multiple_roc_curves(roc_curves: dict):
    fig, axs = plt.subplots(len(roc_curves))
    if len(roc_curves) == 1:
        axs = [axs]
    fig.canvas.set_window_title('Precision Recall Curves')
    fig.suptitle('Precision Recall Curves')

    for i, model_name in enumerate(roc_curves):
        fpr, tpr, thresholds = roc_curves[model_name]
        axs[i].set_title(model_name)
        axs[i].plot(fpr, tpr, linewidth=2)
        axs[i].plot([0, 1], [0, 1], 'k--')
        axs[i].axis([0, 1, 0, 1])
        axs[i].set_xlabel('False Positive Rate')
        axs[i].set_ylabel('True Positive Rate')
    plt.tight_layout()
    plt.show()


def _plot_multiple_conf_matrices(confusion_matrices: dict):
    fig, axs = plt.subplots(len(confusion_matrices))
    if len(confusion_matrices) == 1:
        axs = [axs]
    fig.canvas.set_window_title('Confusion Matrices')
    fig.suptitle('Confusion Matrices')
    for i, model_name in enumerate(confusion_matrices):
        confusion_matrix = confusion_matrices[model_name]
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        normalized_confusion_matrix = confusion_matrix / row_sums
        np.fill_diagonal(normalized_confusion_matrix, 0)
        axs[i].matshow(normalized_confusion_matrix, cmap=plt.cm.get_cmap('gray'))
        axs[i].set_title(model_name)
    plt.tight_layout()
    plt.show()


def _plot_precision_recall(predictions, true_labels, idx):
    scores = [prediction[idx] for prediction in predictions]
    precisions, recalls, thresholds = metrics.precision_recall_curve(true_labels, scores)
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.xlabel('Threshold')
    plt.legend(loc='center left')
    plt.title('Precision Recall Curve')
    fig = plt.gcf()
    fig.canvas.set_window_title('Precision Recall Curve')
    plt.ylim([0, 1])
    plt.show()


def _plot_roc_curve(predictions, true_labels, idx):
    scores = [prediction[idx] for prediction in predictions]
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, scores)
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    fig = plt.gcf()
    fig.canvas.set_window_title('ROC Curve')
    plt.show()


def _plot_confusion_matrix(pred_labels, true_labels):
    confusion_matrix = metrics.confusion_matrix(true_labels, pred_labels)
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    normalized_confusion_matrix = confusion_matrix / row_sums
    np.fill_diagonal(normalized_confusion_matrix, 0)
    plt.matshow(normalized_confusion_matrix, cmap=plt.cm.get_cmap('gray'))
    plt.title('Confusion Matrix')
    fig = plt.gcf()
    fig.canvas.set_window_title('Confusion Matrix')
    plt.show()