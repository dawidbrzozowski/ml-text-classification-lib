from text_clsf_lib.predictors.predictor_commons import get_embedding_preprocessor, get_model, get_tfidf_preprocessor
PRESETS = {

    'tfidf_predictor': {
        'model_path':                              '',
        'preprocessor_func':                        get_tfidf_preprocessor,
        'model_func':                               get_model,
        'preprocessing_params': {
            'text_cleaning_params': {
                'use_ner':                          None,
                'use_ner_converter':                None,
                'use_twitter_data_preprocessing':   None
            },
            'vectorizer_params': {
                'vectorizer_path':                 'preprocessor/vectorizer.vec',
            }
        }
    },


    'embedding_predictor': {
        'model_path':                              '',
        'model_func':                               get_model,
        'preprocessor_func':                        get_embedding_preprocessor,
        'preprocessing_params': {
            'text_cleaning_params': {
                'use_ner':                          None,
                'use_ner_converter':                None,
                'use_twitter_data_preprocessing':   None
            },
            'vectorizer_params': {
                'max_seq_len':                      None,
                'embedding_matrix_path':           'preprocessor/embedding_matrix.npy',
                'text_encoder_path':               'preprocessor/tokenizer.pickle',
            },
        }
    },
}


def create_predictor_preset(
        model_name: str,
        type_: str,  # tfidf or embedding
        model_dir: str = '_models',
        ner_cleaning: bool = False,
        ner_converter: bool = False,
        twitter_preprocessing: bool = False,
        max_seq_len: int = None):
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
    preset = PRESETS['tfidf_predictor'] if type_ == 'tfidf' else PRESETS['embedding_predictor']
    preset['model_path'] = f'{model_dir}/{model_name}/{model_name}.h5'
    preset['preprocessing_params']['text_cleaning_params']['use_ner'] = ner_cleaning
    preset['preprocessing_params']['text_cleaning_params']['use_ner_converter'] = ner_converter
    preset['preprocessing_params']['text_cleaning_params']['use_twitter_data_preprocessing'] = twitter_preprocessing
    if type_ == 'tfidf':
        preset['preprocessing_params']['vectorizer_params']['vectorizer_path'] = \
            f"{model_dir}/{model_name}/{preset['preprocessing_params']['vectorizer_params']['vectorizer_path']}"
    else:
        preset['preprocessing_params']['vectorizer_params']['embedding_matrix_path'] = \
            f'{model_dir}/{model_name}/{preset["preprocessing_params"]["vectorizer_params"]["embedding_matrix_path"]}'
        preset["preprocessing_params"]['vectorizer_params']['text_encoder_path'] = \
            f'{model_dir}/{model_name}/{preset["preprocessing_params"]["vectorizer_params"]["text_encoder_path"]}'
        preset['preprocessing_params']['vectorizer_params']['max_seq_len'] = max_seq_len
    return preset
