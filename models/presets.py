from keras import layers

from data_preparation.data_extracton import BaselineJsonDataExtractor
from models.embedding.components_preparation import prepare_embedding_data_vectorizer
from models.model_builder import TfIdfFFModelBuilder, EmbeddingFFModelBuilder, EmbeddingRNNModelBuilder
from models.tfidf.components_preparation import prepare_tfidf_data_vectorizer
from preprocessing.cleaning.data_cleaners import binary_output
from preprocessing.vectorization.embeddings.embedding_loaders import GloveEmbeddingsLoader
from preprocessing.vectorization.output_vectorizers import BasicOutputVectorizer
from preprocessing.vectorization.text_vectorizers import TfIdfTextVectorizer, EmbeddingTextVectorizer

PRESETS = {

    'tfidf_feedforward': {
        'model_builder_class':                      TfIdfFFModelBuilder,
        'model_name':                              'nn_tfidf',
        'model_save_dir':                          'models/tfidf/_models',
        'data_params': {
            'data_extractor':                       BaselineJsonDataExtractor,
            'cleaning_params': {
                'text': {
                    'use_ner':                          False,
                    'use_ner_converter':                True,
                    'use_twitter_data_preprocessing':   True,
                },
                'output': {
                    'output_verification_func':     binary_output

                }
            },
        },
        'vectorizer_params': {
            'vectorizer_retriever_func':            prepare_tfidf_data_vectorizer,
            'vector_width':                         1000,
            'text_vectorizer':                      TfIdfTextVectorizer,
            'output_vectorizer':                    BasicOutputVectorizer
        },

        'architecture_params': {
            'hidden_layers':                        2,
            'hidden_units':                         32,
            'hidden_activation':                   'relu',
            'output_activation':                   'softmax',
            'optimizer':                           'adam',
            'loss':                                'binary_crossentropy',
            'lr':                                   0.01,
            'metrics':                              ['accuracy'],
            'output_units':                         2
        },
        'training_params': {
            'epochs':                               1,
            'batch_size':                           128,
            'validation_split':                     0.1,
            'callbacks':                            None
        }
    },

    'glove_feedforward': {
        'model_builder_class':                      EmbeddingFFModelBuilder,
        'model_name':                              'nn_embedding',
        'model_save_dir':                          'models/embedding/ff/_models',
        'data_params': {
            'data_extractor':                       BaselineJsonDataExtractor,
            'cleaning_params': {
                'text': {
                    'use_ner':                          False,
                    'use_ner_converter':                True,
                    'use_twitter_data_preprocessing':   True,
                },
                'output': {
                    'output_verification_func':     binary_output

                }
            }
        },
        'vectorizer_params': {
            'vectorizer_retriever_func':            prepare_embedding_data_vectorizer,
            'text_vectorizer':                      EmbeddingTextVectorizer,
            'embeddings_loader':                    GloveEmbeddingsLoader,
            'embedding_type':                      'wiki',
            'output_vectorizer':                    BasicOutputVectorizer,
            'max_vocab_size':                       5000,
            'max_seq_len':                          25,
            'embedding_dim':                        50
        },

        'architecture_params': {
            'dimension_reducer':                    layers.Flatten,
            'hidden_layers':                        2,
            'hidden_units':                         32,
            'hidden_activation':                   'relu',
            'output_activation':                   'softmax',
            'optimizer':                           'adam',
            'loss':                                'binary_crossentropy',
            'lr':                                   0.01,
            'metrics':                              ['accuracy'],
            'trainable_embedding':                  False,
            'output_units':                         2
        },
        'training_params': {
            'epochs':                               5,
            'batch_size':                           128,
            'validation_split':                     0.1,
            'callbacks':                            None
        }
    },

    'glove_rnn': {
        'model_builder_class':                      EmbeddingRNNModelBuilder,
        'model_name':                              'nn_embedding',
        'model_save_dir':                          'models/embedding/rnn/_models',
        'data_params': {
            'data_extractor':                       BaselineJsonDataExtractor,
            'cleaning_params': {
                'text': {
                    'use_ner':                          False,
                    'use_ner_converter':                True,
                    'use_twitter_data_preprocessing':   True,
                },
                'output': {
                    'output_verification_func':     binary_output
                }
            }
        },
        'vectorizer_params': {
            'vectorizer_retriever_func':            prepare_embedding_data_vectorizer,
            'text_vectorizer':                      EmbeddingTextVectorizer,
            'embeddings_loader':                    GloveEmbeddingsLoader,
            'embedding_type':                      'twitter',
            'output_vectorizer':                    BasicOutputVectorizer,
            'max_vocab_size':                       5000,
            'max_seq_len':                          25,
            'embedding_dim':                        50
        },
        'architecture_params': {
            'hidden_layers':                        2,
            'hidden_units':                         32,
            'hidden_activation':                   'relu',
            'output_activation':                   'softmax',
            'optimizer':                           'adam',
            'loss':                                'binary_crossentropy',
            'lr':                                   0.01,
            'metrics':                              ['accuracy'],
            'trainable_embedding':                  False,
            'output_units':                         2
        },
        'training_params': {
            'epochs':                               2,
            'batch_size':                           128,
            'validation_split':                     0.1,
            'callbacks':                            None
        }
    }

}