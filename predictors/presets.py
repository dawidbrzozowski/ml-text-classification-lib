from predictors.predictor_commons import get_embedding_preprocessor, get_model
from preprocessing.cleaning.data_cleaners import BaselineDataCleaner
from preprocessing.vectorization.embeddings.embedding_loaders import GloveEmbeddingsLoader
from preprocessing.vectorization.output_vectorizers import BasicOutputVectorizer
from preprocessing.vectorization.text_vectorizers import EmbeddingTextVectorizer

PRESETS = {
    'glove_predictor': {
        'model_path':                  'models/embedding/ff/_models/nn_embedding.h5',
        'preprocessor_func':            get_embedding_preprocessor,
        'model_func':                   get_model,
        'data_params': {
            'data_cleaner':             BaselineDataCleaner,
            'text_vectorizer':          EmbeddingTextVectorizer,
            'output_vectorizer':        BasicOutputVectorizer,
        },
        'vectorizer_params': {
            'max_vocab_size':           5000,
            'max_seq_len':              25,
            'embedding_dim':            50,
            'embedding_matrix_path':   'preprocessing/_cache/embedding/embedding_matrix.npy',
            'text_encoder_path':       'preprocessing/_cache/embedding/tokenizer.pickle',
            'embedding_loader':         GloveEmbeddingsLoader
        },
    }
}
