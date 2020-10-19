from typing import List
import numpy as np
from sortedcontainers.sortedlist import SortedList

from models.eval.model_prediction import ModelPrediction


def deep_model_test(
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
