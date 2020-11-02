from collections import defaultdict
from typing import List

from text_clsf_lib.models.eval.model_evaluations import deep_samples_test, metrics_test, metrics_test_multiple_models
from text_clsf_lib.models.model_data import prepare_model_data
from text_clsf_lib.models.model_trainer_runner import NNModelRunner
from text_clsf_lib.models.presets.presets_base import PRESETS

DEEP_SAMPLES_DEFAULTS = {
    'show_worst_samples': 10,
}

METRICS_TEST_DEFAULTS = {
    'plot_precision_recall': True,
    'plot_roc_curve': True,
    'plot_conf_matrix': True
}


def test_single_model(
        preset: dict,
        run_deep_samples=False,
        deep_samples_kwargs: dict = None,
        run_metrics_test=False,
        metrics_test_kwargs: dict = None):
    """
    Wrapper function for testing model.
    Enables easy and fast model testing using the preset from training process.
    :param preset: preset used for model training.
    :param run_deep_samples: bool.
    :param deep_samples_kwargs: custom deep_samples_kwargs.
    :param run_metrics_test: bool
    :param metrics_test_kwargs: custom metrics_test_kwargs.
    :return: Predictions of the model and true labels as tuple.
    """
    model_runner = NNModelRunner(model_path=f"{preset['model_save_dir']}/{preset['model_name']}.h5")
    data = prepare_model_data(
        data_params=preset['data_params'],
        vectorizer_params=preset['vectorizer_params'])

    data_test_vec = data['test_vectorized']
    predictions, labels = model_runner.test(data_test_vec)
    if run_deep_samples:
        test_texts, _ = data['test_cleaned']
        if deep_samples_kwargs is not None:
            DEEP_SAMPLES_DEFAULTS.update(deep_samples_kwargs)
        deep_samples_test(test_texts, predictions, labels, **DEEP_SAMPLES_DEFAULTS)
    if run_metrics_test:
        if metrics_test_kwargs is not None:
            METRICS_TEST_DEFAULTS.update(metrics_test_kwargs)
        print(metrics_test(predictions, labels, **METRICS_TEST_DEFAULTS))
    return predictions, labels


def test_multiple_models(presets: List[dict],
                         plot_precision_recall=True,
                         plot_roc_curve=True,
                         plot_conf_matrix=True,
                         plot_model_metrics=True):
    """
    Wrapper function for testing many models at the same time and comparing them.
    Takes list of presets, used during training process.
    Enables showing model comparison plots.
    :param presets: list of preset used during training process for the tested models.
    :param plot_precision_recall: bool
    :param plot_roc_curve: bool
    :param plot_conf_matrix: bool
    :param plot_model_metrics: bool
    :return: dict. Metrics for all models. Metrics include: precision, recall, f1-score, roc_auc_score, confusion_matrix
    """
    model_name_to_scores = defaultdict(dict)
    for preset in presets:
        predictions, labels = test_single_model(preset)
        model_name_to_scores[preset['model_name']]['predictions'] = predictions
        model_name_to_scores[preset['model_name']]['true_labels'] = labels
    return metrics_test_multiple_models(model_name_to_scores,
                                        plot_precision_recall=plot_precision_recall,
                                        plot_roc_curve=plot_roc_curve,
                                        plot_conf_matrix=plot_conf_matrix,
                                        plot_model_metrics=plot_model_metrics)


if __name__ == '__main__':
    presets = PRESETS['glove_rnn']
    test_multiple_models(presets)
