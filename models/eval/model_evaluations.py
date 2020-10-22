from typing import List
import numpy as np
from sortedcontainers.sortedlist import SortedList
from sklearn import metrics
from models.eval.model_prediction import ModelPrediction
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def deep_samples_test(
        texts: List[str],
        predictions: List[np.array],
        true_labels: List[int],
        show_all: bool = False,
        show_worst_samples: int or None = None,
        show_best_samples: int or None = None,
        show_wrong_samples_only: bool = False,
        to_file: None or str = None):
    model_predictions = []
    for text, prediction, true_label in zip(texts, predictions, true_labels):
        model_predictions.append(ModelPrediction(text, prediction, true_label))

    to_file_str = f' to file: {to_file}' if to_file is not None else ''
    if show_all:
        print(f'Showing all samples{to_file_str}...')
        _show_samples(model_predictions, to_file)

    if show_wrong_samples_only:
        print(f'Showing only wrong samples{to_file_str}...')
        wrong_model_predictions = [model_pred for model_pred in model_predictions if not model_pred.is_correct()]
        _show_samples(wrong_model_predictions, to_file)

    if show_worst_samples is not None:
        print(f'Showing {show_worst_samples} worst samples{to_file_str}...')
        worst_predictions = _get_top_n_samples(model_predictions=model_predictions, n=show_worst_samples, best=False)
        _show_samples(worst_predictions, to_file)

    if show_best_samples is not None:
        print(f'Showing {show_best_samples} best samples{to_file_str}...')
        best_predictions = _get_top_n_samples(model_predictions=model_predictions, n=show_best_samples, best=True)
        _show_samples(best_predictions, to_file)


def _show_samples(sample_predictions: List[ModelPrediction], to_file):
    if to_file is not None:
        predictions_str = '\n'.join([str(prediction) for prediction in sample_predictions])
        with open(to_file, 'w') as w_file:
            w_file.write(predictions_str)
    else:
        for prediction in sample_predictions:
            print(prediction)


def _get_top_n_samples(model_predictions: List[ModelPrediction], n: int, best: bool):
    top_n_samples = SortedList(key=lambda sample: -sample.true_label_probability) if best else SortedList()
    for model_prediction in model_predictions:
        if best == model_prediction.is_correct():
            if len(top_n_samples) < n:
                top_n_samples.add(model_prediction)
            else:
                if best != (model_prediction < top_n_samples[-1]):
                    top_n_samples.pop()
                    top_n_samples.add(model_prediction)
    return [sample for sample in top_n_samples]  # so that it returns a normal list instead of SortedList


def metrics_test(predictions, true_labels,
                 plot_precision_recall=False,
                 plot_roc_curve=False,
                 plot_conf_matrix=False) -> dict:
    pred_labels = [np.argmax(prediction) for prediction in predictions]
    if plot_precision_recall:
        _plot_precision_recall(predictions, true_labels, 1)
    if plot_roc_curve:
        _plot_roc_curve(predictions, true_labels, 1)
    if plot_conf_matrix:
        _plot_confusion_matrix(pred_labels, true_labels)
    return {
        'precision': metrics.precision_score(true_labels, pred_labels),
        'recall': metrics.recall_score(true_labels, pred_labels),
        'f1-score': metrics.f1_score(true_labels, pred_labels),
        'roc_auc_score': metrics.roc_auc_score(true_labels, pred_labels),
        'confusion_matrix': metrics.confusion_matrix(true_labels, pred_labels)
    }


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
    normalized_confusion_matrix = confusion_matrix/row_sums
    np.fill_diagonal(normalized_confusion_matrix, 0)
    plt.matshow(normalized_confusion_matrix, cmap=plt.cm.get_cmap('gray'))
    plt.title('Confusion Matrix')
    fig = plt.gcf()
    fig.canvas.set_window_title('Confusion Matrix')
    plt.show()