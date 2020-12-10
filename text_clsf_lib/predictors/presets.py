from text_clsf_lib.predictors.predictor_commons import get_embedding_preprocessor, get_model, get_tfidf_preprocessor, \
    get_bpe_preprocessor


PRESETS = {

    'count_predictor': {
        'preprocessor_func':                        get_tfidf_preprocessor,
        'model_func':                               get_model,
        'preprocessing_params': {
            'config_path':                         'preprocessor/predictor_config.json',
            'vectorizer_params': {
                'vectorizer_path':                 'preprocessor/vectorizer.vec',
            }
        }
    },

    'embedding_predictor': {
        'model_func':                               get_model,
        'preprocessor_func':                        get_embedding_preprocessor,
        'preprocessing_params': {
            'config_path':                         'preprocessor/predictor_config.json',
            'vectorizer_params': {
                'text_encoder_path':               'preprocessor/tokenizer.pickle',
            },
        }
    },

    'bpe_predictor': {
        'model_func':                               get_model,
        'preprocessor_func':                        get_bpe_preprocessor,
        'preprocessing_params': {
                'config_path':                     'preprocessor/predictor_config.json',
        }
    },
}


def create_predictor_preset(
        model_name: str,
        type_: str,  # tfidf or glove or bpe
        model_dir: str = '_models'):
    """
    This function should be used for preparing predictor preset.
    It uses base presets, that are overridden by values provided in arguments.
    :param model_name: str, Your model name. This will be used for determining where is your model located.
    :param type_: str, 'embedding' if the model is based on embeddings or 'tfidf' if the model is based on tfidf.
    :param model_dir: parent directory of a model. Default value: '_models'
    :param ner_cleaning: bool, Whether the NER cleaning should be used on text.
    :param ner_converter: bool, whether the NER cleaning should be converted to proper names,
     better undestood by embedding matrix.
    :param twitter_preprocessing: bool, Use twitter data preprocessing.
    :param max_seq_len: If embedding model is used, here you should specify the max_seq_len.
    :return: dict, predictor preset.
    """
    predictor_presets = {
        'count': 'count_predictor',
        'embedding': 'embedding_predictor',
        'bpe': 'bpe_predictor'
    }
    assert type_ in predictor_presets, f'Type should be one of the following: {[k for k in predictor_presets]}'
    preset = PRESETS[predictor_presets[type_]]
    model_dir = f'{model_dir}/{model_name}'
    model = f'{model_name}.h5'
    preset['preprocessing_params']['model_dir'] = model_dir
    preset['model_path'] = f'{model_dir}/model/{model}'
    return preset
