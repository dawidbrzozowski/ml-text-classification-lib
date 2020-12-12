from text_clsf_lib.models.presets.preset_creation import create_preset, load_preset
from text_clsf_lib.models.build_train_save import train
from semeval_utils.shortcuts import generate_semeval_load_data_args
from text_clsf_lib.models.test_trained_model import test_single_model, test_multiple_models

if __name__ == '__main__':
    data_args = generate_semeval_load_data_args(10000)
    bow_test = create_preset(**data_args,
                             preset_base='bag_of_words_feedforward',
                             model_name='test_new_bow')
    tfidf = create_preset(**data_args,
                             preset_base='tfidf_feedforward',
                             model_name='test_new_tfidf')
    glove = create_preset(**data_args,
                             preset_base='glove_rnn',
                             model_name='test_new_glove')
    bpe = create_preset(**data_args,
                             preset_base='bpe_rnn',
                             model_name='test_new_bpe')

    train(bow_test)
    train(tfidf)
    train(glove)
    train(bpe)

