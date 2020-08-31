from models.model_creator import NNModelRunner
from preprocessing.cleaning.data_cleaners import BaselineDataCleaner
from preprocessing.preprocessors import DataPreprocessor
from preprocessing.vectorization.data_vectorizers import DataVectorizer
from preprocessing.vectorization.embeddings.embedding_loaders import GloveEmbeddingsLoader
from preprocessing.vectorization.embeddings.text_encoders import TextEncoder
from preprocessing.vectorization.output_vectorizers import BasicOutputVectorizer
from preprocessing.vectorization.text_vectorizers import EmbeddingTextVectorizer, TfIdfTextVectorizer
from utils.files_io import read_pickle, read_numpy


class PredictorPreparer:
    def __init__(self, config):
        preprocessing_config = config.get('preprocessor')
        data_cleaner = self._prepare_data_cleaner(preprocessing_config.get('data_cleaner'))
        data_vectorizer = self._prepare_data_vectorizer(preprocessing_config.get('data_vectorizer'))
        self.data_preprocessor = DataPreprocessor(data_cleaner, data_vectorizer)
        self.model_runner = NNModelRunner(model_path=config['model_path'])

    def get_preprocessor(self):
        return self.data_preprocessor

    def get_model_runner(self):
        return self.model_runner

    def _prepare_data_cleaner(self, data_cleaner: dict):
        data_cleaners = {
            'BaselineDataCleaner': BaselineDataCleaner()
        }
        retrieved_obj = data_cleaners.get(data_cleaner['name'])
        assert retrieved_obj is not None, 'Wrong DataCleaner name!'
        return retrieved_obj

    def _prepare_data_vectorizer(self, data_vectorizer: dict):
        text_vectorizer = self._prepare_text_vectorizer(data_vectorizer.get('text_vectorizer'))
        output_vectorizer = self._prepare_output_vectorizer(data_vectorizer.get('output_vectorizer'))
        return DataVectorizer(text_vectorizer, output_vectorizer)

    def _prepare_text_vectorizer(self, text_vectorizer: dict):
        if text_vectorizer['type'] == 'embedding':
            return self._prepare_embedding_text_vectorizer(text_vectorizer)
        elif text_vectorizer['type'] == 'tfidf':
            return self.prepare_tfidf_text_vectorizer(text_vectorizer)

    def _prepare_output_vectorizer(self, output_vectorizer: dict):
        output_vectorizers = {
            'BasicOutputVectorizer': BasicOutputVectorizer()
        }
        retrieved_obj = output_vectorizers.get(output_vectorizer['name'])
        assert retrieved_obj is not None, 'Wrong OutputVectorizer name!'
        return retrieved_obj

    def _prepare_embedding_text_vectorizer(self, text_vectorizer: dict):
        embeddings_loaders = {
            'glove': GloveEmbeddingsLoader()
        }
        embeddings_data = text_vectorizer.get('embeddings')
        embeddings_loader = embeddings_loaders.get(embeddings_data['embedding_type'])
        assert embeddings_loader is not None, 'Wrong embedding type name!'
        text_encoder = self._prepare_text_encoder(embeddings_data.get('text_encoder'))
        embedding_dim = embeddings_data['embedding_dim']
        embedding_matrix = read_numpy(embeddings_data['embedding_matrix_path'])
        embedding_text_vectorizers = {
            'EmbeddingTextVectorizer': EmbeddingTextVectorizer(text_encoder, embedding_dim, embeddings_loader,
                                                               embedding_matrix)
        }
        retrieved_obj = embedding_text_vectorizers.get(text_vectorizer['name'])
        assert retrieved_obj is not None, 'Wrong Embedding Text Vectorizer name!'
        return retrieved_obj

    def _prepare_text_encoder(self, text_encoder: dict):
        max_vocab_size = text_encoder['max_vocab_size']
        max_seq_len = text_encoder['max_seq_len']
        tokenizer = read_pickle(text_encoder['text_encoder_path'])
        text_encoders = {
            'TextEncoder': TextEncoder(max_vocab_size, max_seq_len, tokenizer)
        }
        retrieved_obj = text_encoders.get(text_encoder['name'])
        assert retrieved_obj is not None, 'Wrong Text Encoder name!'
        return retrieved_obj

    def prepare_tfidf_text_vectorizer(self, text_vectorizer):
        max_features = text_vectorizer['max_features']
        vectorizer = read_pickle(text_vectorizer['tfidf_vec_path'])
        tfidf_vectorizers = {
            'TfIdfTextVectorizer': TfIdfTextVectorizer(max_features, vectorizer)
        }
        retrieved_obj = tfidf_vectorizers.get(text_vectorizer['name'])
        assert retrieved_obj is not None, 'Wrong TfIdf Text Vectorizer name!'
        return retrieved_obj
