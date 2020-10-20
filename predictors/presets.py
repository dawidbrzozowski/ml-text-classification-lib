from predictors.predictor_commons import get_embedding_preprocessor, get_model, get_tfidf_preprocessor
from preprocessing.cleaning.data_cleaners import BaselineDataCleaner
from preprocessing.vectorization.text_vectorizers import LoadedEmbeddingTextVectorizer, LoadedTfIdfTextVectorizer

PRESETS = {

    'glove_ff_predictor': {
        'model_path':                               'models/embedding/ff/_models/nn_embedding.h5',
        'preprocessor_func':                         get_embedding_preprocessor,
        'model_func':                                get_model,
        'data_params': {
            'text_cleaning_params': {
                'use_ner':                           False,
                'use_ner_converter':                 True,
                'use_twitter_data_preprocessing':    True
            },
            'text_vectorizer':                       LoadedEmbeddingTextVectorizer,
        },
        'vectorizer_params': {
            'max_seq_len':                          25,
            'embedding_dim':                        50,
            'embedding_matrix_path':               'preprocessing/_cache/embedding/embedding_matrix.npy',
            'text_encoder_path':                   'preprocessing/_cache/embedding/tokenizer.pickle',
        },
    },


    'tfidf_predictor': {
        'model_path':                              'models/tfidf/_models/nn_tfidf.h5',
        'preprocessor_func':                        get_tfidf_preprocessor,
        'model_func':                               get_model,
        'data_params': {
            'text_cleaning_params': {
                'use_ner':                          False,
                'use_ner_converter':                True,
                'use_twitter_data_preprocessing':   True
            },
            'text_vectorizer':                      LoadedTfIdfTextVectorizer,
        },
        'vectorizer_params': {
            'vectorizer_path':                     'preprocessing/_cache/tfidf/vectorizer.vec',
            'vector_width':                         1000

        },
    },


    'glove_rnn_predictor': {
        'model_path':                              'models/embedding/rnn/_models/nn_embedding.h5',
        'preprocessor_func':                        get_embedding_preprocessor,
        'model_func':                               get_model,
        'data_params': {
            'text_cleaning_params': {
                'use_ner':                          False,
                'use_ner_converter':                True,
                'use_twitter_data_preprocessing':   True
            },
            'text_vectorizer':                      LoadedEmbeddingTextVectorizer,
        },
        'vectorizer_params': {
            'max_seq_len':                          25,
            'embedding_dim':                        50,
            'embedding_matrix_path':               'preprocessing/_cache/embedding/embedding_matrix.npy',
            'text_encoder_path':                   'preprocessing/_cache/embedding/tokenizer.pickle',
        },
    },
}
