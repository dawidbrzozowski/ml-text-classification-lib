from models.model_trainer_runner import NNModelRunner
from models.presets import PRESETS


def test(preset: dict):
    model_runner = NNModelRunner(model_path=f"{preset['model_save_dir']}/{preset['model_name']}.h5")
    data_func = preset['data_func']
    data_params = preset['data_params']
    vectorizer_params = preset['vectorizer_params']
    data_train, data_test = data_func(data_params=data_params, vectorizer_params=vectorizer_params)
    model_runner.test(data_test)


if __name__ == '__main__':
    preset_name = 'glove_rnn'
    test(PRESETS[preset_name])