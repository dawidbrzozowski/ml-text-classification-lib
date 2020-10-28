from text_clsf_lib.models.presets.preset_creation import create_preset
from text_clsf_lib.models.build_train_save import train
from text_clsf_lib.models.test_trained_model import test_single_model
if __name__ == '__main__':
    preset = create_preset('glove_rnn',
                           model_name='tfidf_custom',
                           hidden_layers_list=['bidirectional',
                                               'lstm tanh 12 return_sequences',
                                               'globalmaxpooling1d',
                                               'dense relu 12',
                                               'dense softmax 2'],
                           twitter_preprocessing=False,
                           epochs=2)
    model_runner = train(preset)
    # print(test_single_model(preset, run_metrics_test=True))