from keras import layers

from text_clsf_lib.data_preparation.data_extracton import BaselineJsonDataExtractor
from text_clsf_lib.models.embedding.components_preparation import prepare_embedding_data_vectorizer
from text_clsf_lib.models.build.model_builder import TfIdfFFModelBuilder, EmbeddingFFModelBuilder, EmbeddingRNNModelBuilder
from text_clsf_lib.models.tfidf.components_preparation import prepare_tfidf_data_vectorizer
from text_clsf_lib.preprocessing.cleaning.data_cleaners import binary_output
from text_clsf_lib.preprocessing.vectorization.embeddings.embedding_loaders import GloveEmbeddingsLoader
from text_clsf_lib.preprocessing.vectorization.output_vectorizers import BasicOutputVectorizer
from text_clsf_lib.preprocessing.vectorization.text_vectorizers import TfIdfTextVectorizer, EmbeddingTextVectorizer, \
    BagOfWordsTextVectorizer

PRESETS = {

    'tfidf_feedforward': {
        'model_builder_class':                      TfIdfFFModelBuilder,
        'model_name':                              'nn_tfidf',
        'model_save_dir':                          '_models',
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
            'save_dir':                            'preprocessor',
            'vector_width':                         1000,
            'text_vectorizer':                      TfIdfTextVectorizer,
            'output_vectorizer':                    BasicOutputVectorizer
        },

        'architecture_params': {
            'hidden_layers_list':                   [],
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
    'bag_of_words_feedforward': {
        'model_builder_class':                      TfIdfFFModelBuilder,
        'model_name':                              'nn_tfidf',
        'model_save_dir':                          '_models',
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
            'save_dir':                            'preprocessor',
            'vector_width':                         1000,
            'text_vectorizer':                      BagOfWordsTextVectorizer,
            'output_vectorizer':                    BasicOutputVectorizer
        },

        'architecture_params': {
            'hidden_layers_list':                   [],
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
        'model_save_dir':                          '_models',
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
            'embedding_dim':                        50,
            'save_dir':                            'preprocessor',
        },

        'architecture_params': {
            'hidden_layers_list':                   [],
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
        'model_save_dir':                          '_models',
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
            'max_seq_len':                          200,
            'embedding_dim':                        50,
            'save_dir':                            'preprocessor',

        },
        'architecture_params': {
            'hidden_layers_list':                   [],
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