from text_clsf_lib.models.model_data import prepare_model_data
from text_clsf_lib.models.model_trainer_runner import NNModelTrainer, NNModelRunner
from text_clsf_lib.models.presets.presets_base import PRESETS


def train(preset: dict,
          save_tflite_model=False):
    data = prepare_model_data(
        data_params=preset['data_params'],
        vectorizer_params=preset['vectorizer_params'])

    model_trainer = get_model_trainer(preset)

    data_train = data['train_vectorized']

    model_trainer.train(data_train)
    print('Training process complete!')
    model_trainer.save(preset['model_save_dir'], preset['model_name'])

    if save_tflite_model:
        model_trainer.save_tflite(preset['model_save_dir'], preset['model_name'])
    print(f'Model saved to: {preset["model_save_dir"]}. Model name: {preset["model_name"]}')
    return NNModelRunner(model=model_trainer.model)


def get_model_trainer(preset: dict):
    return NNModelTrainer(
        model_builder_class=preset['model_builder_class'],
        architecture_params=preset['architecture_params'],
        vectorizer_params=preset['vectorizer_params'],
        training_params=preset['training_params']
    )


if __name__ == '__main__':
    preset_name = 'glove_rnn'

    train(PRESETS[preset_name])
