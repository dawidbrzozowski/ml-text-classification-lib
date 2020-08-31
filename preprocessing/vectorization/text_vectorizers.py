from abc import abstractmethod
from typing import List

import os
from preprocessing.vectorization.embeddings.embedding_loaders import GloveEmbeddingsLoader
from preprocessing.vectorization.embeddings.embeddings import EmbeddingsMatrixPreparer
from preprocessing.vectorization.embeddings.text_encoders import TextEncoderBase
from utils.files_io import write_pickle, write_numpy
from project_settings import PREPROCESSING_SAVE_DIR as SAVE_DIR

VECTORIZER_NAME = 'vectorizer.vec'
EMBEDDING_MATRIX_NAME = 'embedding_matrix.npy'


class TextVectorizer:
    """
    This method is meant to perform all text vectorization.
    TODO it should somehow tell if embedding layer is needed or not.
    TODO In this case TfIdfTextVectorizer does not need embedding layer but EmbeddingTextVectorizer does.
    TODO That's why EmbeddingTextVectorizer should also offer get_embedding_matrix method.
    """

    @abstractmethod
    def fit(self, texts: List[str]):
        pass

    @abstractmethod
    def vectorize(self, texts: List[str]):
        pass

    @abstractmethod
    def get_vectorization_metainf(self):
        pass


class TfIdfTextVectorizer(TextVectorizer):
    def __init__(self, max_features, vectorizer=None):
        self.tfidf_vec = vectorizer if vectorizer is not None else TfIdfTextVectorizer(max_features=max_features)

    def fit(self, texts: List[str]):
        self.tfidf_vec.fit(texts)
        os.makedirs(f'{SAVE_DIR}/tfidf', exist_ok=True)
        write_pickle(f'{SAVE_DIR}/tfidf/{VECTORIZER_NAME}', self.tfidf_vec)

    def vectorize(self, texts: List[str]):
        return self.tfidf_vec.transform(texts).toarray()

    def get_vectorization_metainf(self):
        return {
            'type': 'tfidf',
            'max_features': self.tfidf_vec.max_features,
            'vocab': self.tfidf_vec.vocabulary_
        }


class EmbeddingTextVectorizer(TextVectorizer):
    def __init__(self, text_encoder: TextEncoderBase, embedding_dim, embeddings_loader=GloveEmbeddingsLoader()):
        self.text_encoder = text_encoder
        self.embedding_dim = embedding_dim
        self.embedding_matrix = None
        self.embeddings_loader = embeddings_loader

    def fit(self, texts: List[str]):
        self.text_encoder.fit(texts)
        word2vec = self.embeddings_loader.load_word_vectors(self.embedding_dim)
        emb_matrix_preparer = EmbeddingsMatrixPreparer(self.text_encoder.word2idx, word2vec)
        self.embedding_matrix = emb_matrix_preparer.prepare_embedding_matrix()
        os.makedirs(f'{SAVE_DIR}/embedding', exist_ok=True)
        write_numpy(f'{SAVE_DIR}/embedding/{EMBEDDING_MATRIX_NAME}', self.embedding_matrix)

    def vectorize(self, texts: List[str]):
        return self.text_encoder.encode(texts)

    def get_vectorization_metainf(self):
        return {
            'type': 'embedding',
            'embedding_dim': self.embedding_dim,
            'embedding_matrix': self.embedding_matrix,
            'max_seq_len': self.text_encoder.max_seq_len,
            'max_vocab_size': self.text_encoder.max_vocab_size
        }
