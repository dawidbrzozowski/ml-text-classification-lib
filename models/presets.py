from keras import layers

from data_preparation.data_extracton import BaselineJsonDataExtractor
from models.model_builder import TfIdfFFModelBuilder, EmbeddingFFModelBuilder, EmbeddingRNNModelBuilder
from preprocessing.cleaning.data_cleaners import BaselineDataCleaner
from preprocessing.vectorization.output_vectorizers import BasicOutputVectorizer
from preprocessing.vectorization.text_vectorizers import TfIdfTextVectorizer, EmbeddingTextVectorizer
from models.tfidf.model_data import prepare_tfidf_train_test_data
from models.embedding.model_data import prepare_embedding_train_test_data

PRESETS = {
    'tfidf_feedforward': {
        'model_builder_class':      TfIdfFFModelBuilder,
        'data_func':                prepare_tfidf_train_test_data,
        'data_params': {
            'data_extractor':       BaselineJsonDataExtractor,
            'data_cleaner':         BaselineDataCleaner,
            'text_vectorizer':      TfIdfTextVectorizer,
            'output_vectorizer':    BasicOutputVectorizer,
        },
        'model_name':              'nn_tfidf',
        'model_save_dir':          'models/tfidf/_models',
        'vectorizer_params': {
            'vector_width':         1000
        },
        'architecture_params': {
            'hidden_layers':        2,
            'hidden_units':         32,
            'hidden_activation':   'relu',
            'output_activation':   'sigmoid',
            'optimizer':           'adam',
            'loss':                'binary_crossentropy',
            'lr':                   0.01,
            'metrics':              ['accuracy']
        },
        'training_params': {
            'epochs':               5,
            'batch_size':           128,
            'validation_split':     0.1,
            'callbacks':            None
        }
    },

    'glove_feedforward': {
        'model_builder_class':      EmbeddingFFModelBuilder,
        'data_func':                prepare_embedding_train_test_data,
        'data_params': {
            'data_extractor':       BaselineJsonDataExtractor,
            'data_cleaner':         BaselineDataCleaner,
            'text_vectorizer':      EmbeddingTextVectorizer,
            'output_vectorizer':    BasicOutputVectorizer,
        },
        'model_name':              'nn_embedding',
        'model_save_dir':          'models/embedding/ff/_models',
        'vectorizer_params': {
            'max_vocab_size':       5000,
            'max_seq_len':          25,
            'embedding_dim':        50
        },
        'architecture_params': {
            'dimension_reducer':    layers.Flatten,
            'hidden_layers':        2,
            'hidden_units':         32,
            'hidden_activation':   'relu',
            'output_activation':   'sigmoid',
            'optimizer':           'adam',
            'loss':                'binary_crossentropy',
            'lr':                   0.01,
            'metrics':              ['accuracy'],
            'trainable_embedding':  False
        },
        'training_params': {
            'epochs':               5,
            'batch_size':           128,
            'validation_split':     0.1,
            'callbacks':            None
        }
    },

    'glove_rnn': {
        'model_builder_class':      EmbeddingRNNModelBuilder,
        'data_func':                prepare_embedding_train_test_data,
        'data_params': {
            'data_extractor':       BaselineJsonDataExtractor,
            'data_cleaner':         BaselineDataCleaner,
            'text_vectorizer':      EmbeddingTextVectorizer,
            'output_vectorizer':    BasicOutputVectorizer,
        },
        'model_name':              'nn_embedding',
        'model_save_dir':          'models/embedding/rnn/_models',
        'vectorizer_params': {
            'max_vocab_size':       5000,
            'max_seq_len':          25,
            'embedding_dim':        50
        },
        'architecture_params': {
            'hidden_layers':        2,
            'hidden_units':         32,
            'hidden_activation':   'relu',
            'output_activation':   'sigmoid',
            'optimizer':           'adam',
            'loss':                'binary_crossentropy',
            'lr':                   0.01,
            'metrics':              ['accuracy'],
            'trainable_embedding':  False
        },
        'training_params': {
            'epochs':               5,
            'batch_size':           128,
            'validation_split':     0.1,
            'callbacks':            None
        }
    }
}
