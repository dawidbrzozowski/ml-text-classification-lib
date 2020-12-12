from text_clsf_lib.models.presets.preset_creation import create_preset, load_preset
from text_clsf_lib.models.build_train_save import train
from semeval_utils.shortcuts import generate_semeval_load_data_args
from text_clsf_lib.models.test_trained_model import test_single_model, test_multiple_models

if __name__ == '__main__':
    bow = load_preset('bow_best')
    bpe_best_model = load_preset('bpe_best_model')
    #glove_ff_twitter = load_preset('glove_ff_twitter')
    #glove_ff_wiki = load_preset('glove_ff_wiki')
    glove_rnn_twitter = load_preset('glove_rnn_twitter')
    #glove_rnn_wiki = load_preset('glove_rnn_wiki')
    tfidf_best = load_preset('tfidf_best')
    test_multiple_models([bow, bpe_best_model, glove_rnn_twitter, tfidf_best])

