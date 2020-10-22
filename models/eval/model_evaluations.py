from collections import defaultdict
from typing import List
import numpy as np
from sortedcontainers.sortedlist import SortedList
from sklearn import metrics
from models.eval.model_prediction import ModelPrediction
import matplotlib

from models.eval.plots import _plot_multiple_precision_recall_curves, _plot_multiple_roc_curves, \
    _plot_multiple_conf_matrices, _plot_precision_recall, _plot_roc_curve, _plot_confusion_matrix

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


def metrics_test_multiple_models(model_output_true_label: dict,
                                 plot_precision_recall=False,
                                 plot_roc_curve=False,
                                 plot_conf_matrix=False) -> dict:
    model_metrics = defaultdict(dict)
    model_curves = defaultdict(dict)
    for model_name in model_output_true_label:
        true_labels = model_output_true_label[model_name]['true_labels']
        predictions = model_output_true_label[model_name]['predictions']
        offensive_predictions = [prediction[1] for prediction in predictions]
        pred_labels = [np.argmax(prediction) for prediction in predictions]
        if plot_precision_recall:
            model_curves['precision_recall'][model_name] = metrics.precision_recall_curve(true_labels,
                                                                                          offensive_predictions)
        if plot_roc_curve:
            model_curves['roc'][model_name] = metrics.roc_curve(true_labels, offensive_predictions)

        if plot_conf_matrix:
            model_curves['confusion_matrix'][model_name] = metrics.confusion_matrix(true_labels, pred_labels)

        model_metrics['precision'][model_name] = metrics.precision_score(true_labels, pred_labels)
        model_metrics['recall'][model_name] = metrics.recall_score(true_labels, pred_labels)
        model_metrics['f1_score'][model_name] = metrics.f1_score(true_labels, pred_labels)
        model_metrics['roc_auc_score'][model_name] = metrics.roc_auc_score(true_labels, pred_labels)
        model_metrics['confusion_matrix'][model_name] = metrics.confusion_matrix(true_labels, pred_labels)

    if plot_precision_recall:
        _plot_multiple_precision_recall_curves(model_curves['precision_recall'])

    if plot_roc_curve:
        _plot_multiple_roc_curves(model_curves['roc'])

    if plot_conf_matrix:
        _plot_multiple_conf_matrices(model_curves['confusion_matrix'])

    return model_metrics


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