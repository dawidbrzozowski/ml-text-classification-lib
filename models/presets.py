from data_preparation.data_extracton import BaselineDataExtractor
from preprocessing.cleaning.data_cleaners import BaselineDataCleaner
from preprocessing.vectorization.output_vectorizers import BasicOutputVectorizer
from preprocessing.vectorization.text_vectorizers import TfIdfTextVectorizer, EmbeddingTextVectorizer
from models.tfidf.model_commons import prepare_train_test_data as prepare_tfidf_data
PRESETS = {
    'tfidf_feedforward': {
        'data_func':                prepare_tfidf_data,
        'data_params': {
            'data_extractor':       BaselineDataExtractor,
            'data_cleaner':         BaselineDataCleaner,
            'text_vectorizer':      TfIdfTextVectorizer,
            'output_vectorizer':    BasicOutputVectorizer,
        },
        'vectorizer_params': {
            'vector_width':         1000
        },
        'training_params': {
            'epochs':               5,
            'hidden_layers':        2,
            'hidden_units':         32,
            'batch_size':           128,
            'hidden_activation':   'relu',
            'output_activation':   'sigmoid',
            'optimizer':           'adam',
            'loss':                'default_loss',
            'validation_split':     0.1,
            'metrics':              ['accuracy'],
            'callbacks':            None,
            'lr':                   0.01
            }
        },

    'glove_feedforward': {
        'data_extractor':               BaselineDataExtractor,
        'data_cleaner':                 BaselineDataCleaner,
        'text_vectorizer':              EmbeddingTextVectorizer,
        'output_vectorizer':            BasicOutputVectorizer,
        'vectorizer_params': {
            'max_vocab_size':           5000,
            'max_seq_len':              25,
            'embedding_dim':            50,
        },
        'training_params': {
            'epochs':                   5,
            'hidden_layers':            2,
            'hidden_units':             32,
            'batch_size':               128,
            'hidden_activation':       'relu',
            'output_activation':       'sigmoid',
            'optimizer':               'adam',
            'loss':                    'default_loss',
            'validation_split':         0.1,
            'metrics':                  ['accuracy'],
            'callbacks':                None,
            'lr':                       0.01
        }
    },

    'glove_rnn': None
}
