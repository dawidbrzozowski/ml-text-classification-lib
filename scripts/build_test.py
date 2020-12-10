from text_clsf_lib.models.presets.preset_creation import create_preset
from text_clsf_lib.models.build_train_save import train
from semeval_utils.corp_functions import generate_semeval_load_data_args
if __name__ == '__main__':
    data_args = generate_semeval_load_data_args(100000)

    my_bow = create_preset(preset_base='bpe_rnn',
                           use_corpus_balancing=True,
                           **data_args)
    train(my_bow)
    # test_single_model_sample_analysis(bpe_rnn, )

