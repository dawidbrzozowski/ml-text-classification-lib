from text_clsf_lib.models.presets.preset_creation import create_preset, load_preset
from text_clsf_lib.models.build_train_save import train
from text_clsf_lib.models.test_trained_model import test_single_model, test_multiple_models, test_single_model_sample_analysis

if __name__ == '__main__':
    preset_glove_rnn = create_preset('glove_rnn',
                                     model_name='glove_rnn',
                                     twitter_preprocessing=True,
                                     epochs=1,
                                     lr=0.001,
                                     ner_cleaning=False)
    my_tfidf = load_preset('my_tfidf')
    my_bow = create_preset('bag_of_words_feedforward',
                           model_name='my_bow',
                           twitter_preprocessing=False)
    bpe_rnn = create_preset('bpe_rnn',
                            model_name='bpe_current_best',
                            twitter_preprocessing=True,
                            use_lowercase=True,
                            max_seq_len=200,
                            use_corpus_balancing=True,
                            y_name='offensive',
                            hidden_layers_list=[
                              'bidirectional lstm 100 return_sequences tanh',
                              'globalmaxpooling1d',
                              'dense relu 50',
                              'dense softmax 2'
                            ],
                            epochs=4,
                            lr=0.001)
    test_single_model(my_tfidf)
    # test_single_model_sample_analysis(bpe_rnn, )
