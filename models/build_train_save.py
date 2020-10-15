from models.model_trainer_runner import NNModelTrainer
from models.presets import PRESETS


def train(preset: dict):
    data_func = preset['data_func']
    data_params = preset['data_params']
    vectorizer_params = preset['vectorizer_params']
    data_train, data_test = data_func(data_params=data_params, vectorizer_params=vectorizer_params)
    model_trainer = NNModelTrainer(preset)
    model_trainer.train(data_train)
    print('Training process complete!')
    model_trainer.save(preset['model_save_dir'], preset['model_name'])
    print(f'Model saved to: {preset["model_save_dir"]}. Model name: {preset["model_name"]}')


if __name__ == '__main__':
    preset_name = 'glove_rnn'

    train(PRESETS[preset_name])
