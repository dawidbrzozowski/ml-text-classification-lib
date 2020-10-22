from models.eval.model_evaluations import deep_samples_test, metrics_test, metrics_test_multiple_models
from models.model_trainer_runner import NNModelRunner
from models.presets import PRESETS


def test(preset: dict):
    model_runner = NNModelRunner(model_path=f"{preset['model_save_dir']}/{preset['model_name']}.h5")
    data_func = preset['data_func']
    data_params = preset['data_params']
    vectorizer_params = preset['vectorizer_params']
    data = data_func(data_params=data_params, vectorizer_params=vectorizer_params)
    data_test_vec = data['test_vectorized']
    results = model_runner.test(data_test_vec)
    test_texts, _ = data['test_cleaned']
    # deep_samples_test(test_texts, results['predictions'], results['labels'],
    # show_worst_samples=10, to_file='worst_samples.txt')
    # print(metrics_test(results['predictions'], results['labels'], True, True, True))

    print(metrics_test_multiple_models(
        {
            'glove_rnn': {'predictions': results['predictions'], 'true_labels': results['labels']},
            'glove_rnn2': {'predictions': results['predictions'], 'true_labels': results['labels']}
        },
        True, True, True))


if __name__ == '__main__':
    preset_name = 'glove_rnn'
    test(PRESETS[preset_name])
