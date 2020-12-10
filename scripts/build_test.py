from text_clsf_lib.models.presets.preset_creation import create_preset
from text_clsf_lib.models.build_train_save import train
from semeval_utils.corp_functions import generate_semeval_load_data_args
if __name__ == '__main__':
    data_args = generate_semeval_load_data_args(6000000)

    rnn_bpe = create_preset(preset_base='bpe_rnn',
                           use_corpus_balancing=True,
                           model_name="bpe_clean_branch",
                           corpus_word_limit=130,
                           twitter_preprocessing=True,
                           use_lowercase=True,
                           max_seq_len=200,
                           epochs=6,
                           lr=0.001,
                           hidden_layers_list = [
                              "bidirectional lstm 100 return_sequences tanh",
                              "spatialdropout1d 0.2",
                              "bidirectional lstm 100 return_sequences tanh",
                              "globalmaxpooling1d",
                              "dense relu 50",
                              "dropout1d 0.2",
                              "dense relu 50",
                              "dense softmax 2"],
                           **data_args)
    train(rnn_bpe)
    # test_single_model_sample_analysis(bpe_rnn, )
