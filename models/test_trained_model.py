from models.eval.model_evaluations import deep_model_test
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
    deep_model_test(test_texts, results['predictions'], results['labels'],
                    show_worst_samples=10, to_file='worst_samples.txt')


if __name__ == '__main__':
    preset_name = 'tfidf_feedforward'
    test(PRESETS[preset_name])