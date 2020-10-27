from text_clsf_lib.models.presets.preset_creation import create_preset
from text_clsf_lib.models.build_train_save import train
from text_clsf_lib.models.test_trained_model import test_single_model
if __name__ == '__main__':
    preset = create_preset('glove_rnn',
                           model_name='glove_rnn',
                           twitter_preprocessing=True,
                           epochs=2)
    model_runner = train(preset)
    # print(test_single_model(preset, run_metrics_test=True))