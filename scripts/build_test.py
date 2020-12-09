from text_clsf_lib.models.presets.preset_creation import create_preset
from text_clsf_lib.models.build_train_save import train
from text_clsf_lib.models.test_trained_model import test_single_model, test_multiple_models

# TODO generować preset w folderze z modelem
if __name__ == '__main__':
    preset_glove_ff = create_preset('glove_feedforward',
                                    model_name='glove_feedforward',
                                    twitter_preprocessing=False,
                                    epochs=2)
    preset_glove_rnn = create_preset('glove_rnn',
                                     model_name='glove_rnn',
                                     twitter_preprocessing=True,
                                     epochs=1,
                                     lr=0.001,
                                     ner_cleaning=False)
    my_tfidf = create_preset('tfidf_feedforward',
                             model_name='my_tfidf',
                             twitter_preprocessing=False)
    my_bow = create_preset('bag_of_words_feedforward',
                           model_name='my_bow',
                           twitter_preprocessing=False)
    bpe_rnn = create_preset('bpe_rnn',
                            model_name='bpe_rnn',
                            twitter_preprocessing=True,
                            use_lowercase=True,
                            epochs=1,
                            lr=0.001,
                            ner_cleaning=False)
    print(test_multiple_models([bpe_rnn]))
