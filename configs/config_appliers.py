from preprocessing.cleaning.data_cleaners import BaselineDataCleaner
from preprocessing.preprocessors import DataPreprocessor
from preprocessing.vectorization.data_vectorizers import DataVectorizer
from preprocessing.vectorization.embeddings.embedding_loaders import GloveEmbeddingsLoader
from preprocessing.vectorization.embeddings.text_encoders import TextEncoder
from preprocessing.vectorization.output_vectorizers import BasicOutputVectorizer
from preprocessing.vectorization.text_vectorizers import EmbeddingTextVectorizer
from utils.files_io import load_json


class Config:
    def __init__(self, config_fp):
        self.configs = load_json(config_fp)

    def get_preprocessor(self):
        data_cleaner = BaselineDataCleaner() # no other choices for now
        text_vectorizer = None
        output_vectorizer = None
        if self.configs['preprocessor']['data_vectorizer']['text_vectorizer']['type'] == 'embedding':
            embeddings = self.configs['preprocessor']['data_vectorizer']['text_vectorizer'].get('embeddings')
            embedding_dim = embeddings['embedding_dim']
            embeddings_loader = None
            if embeddings['embedding_type'] == 'glove':
                embeddings_loader = GloveEmbeddingsLoader()
            max_vocab_size = embeddings['text_encoder']['max_vocab_size']
            max_seq_len = embeddings['text_encoder']['max_seq_len']
            text_encoder = None
            if embeddings['text_encoder']['name'] == 'TextEncoder':
                text_encoder = TextEncoder(max_vocab_size, max_seq_len)
            if self.configs['preprocessor']['data_vectorizer']['text_vectorizer']['name'] == 'EmbeddingTextVectorizer':
                text_vectorizer = EmbeddingTextVectorizer(text_encoder, embedding_dim, embeddings_loader)

        if self.configs['preprocessor']['data_vectorizer']['output_vectorizer']['name'] == 'BasicOutputVectorizer':
            output_vectorizer = BasicOutputVectorizer()
        data_vectorizer = DataVectorizer(text_vectorizer, output_vectorizer)

        return DataPreprocessor(data_cleaner, data_vectorizer)
