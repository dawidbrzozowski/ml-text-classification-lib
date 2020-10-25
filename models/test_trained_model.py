from collections import defaultdict
from typing import List

from models.eval.model_evaluations import deep_samples_test, metrics_test, metrics_test_multiple_models
from models.model_trainer_runner import NNModelRunner
from models.presets import PRESETS


def test_single_model(preset_name: str, verb=0):
    preset = PRESETS[preset_name]
    model_runner = NNModelRunner(model_path=f"{preset['model_save_dir']}/{preset['model_name']}.h5")
    data_func = preset['data_func']
    data_params = preset['data_params']
    vectorizer_params = preset['vectorizer_params']
    data = data_func(data_params=data_params, vectorizer_params=vectorizer_params)
    data_test_vec = data['test_vectorized']
    predictions, labels = model_runner.test(data_test_vec)
    if verb > 0:
        test_texts, _ = data['test_cleaned']
        deep_samples_test(test_texts, predictions, labels, show_worst_samples=10, to_file='worst_samples.txt')
        print(metrics_test(predictions, labels, True, True, True))
    return predictions, labels


def test_multiple_models(preset_names: List[str]):
    model_name_to_scores = defaultdict(dict)
    for preset_name in preset_names:
        predictions, labels = test_single_model(preset_name)
        model_name_to_scores[preset_name]['predictions'] = predictions
        model_name_to_scores[preset_name]['true_labels'] = labels
    metrics_test_multiple_models(model_name_to_scores, True, True, True, True)


if __name__ == '__main__':
    preset_names = ['glove_rnn', 'tfidf_feedforward', 'glove_feedforward']
    test_multiple_models(preset_names)
