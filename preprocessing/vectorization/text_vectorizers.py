from abc import abstractmethod
from typing import List

import os

from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing.vectorization.embeddings.embedding_loaders import EmbeddingsLoader
from preprocessing.vectorization.embeddings.embeddings import EmbeddingsMatrixPreparer
from preprocessing.vectorization.embeddings.text_encoders import TextEncoderBase, LoadedTextEncoder
from utils.files_io import write_pickle, write_numpy
from project_settings import PREPROCESSING_SAVE_DIR as SAVE_DIR

VECTORIZER_NAME = 'vectorizer.vec'
EMBEDDING_MATRIX_NAME = 'embedding_matrix.npy'


class TextVectorizer:

    @abstractmethod
    def fit(self, texts: List[str]):
        pass

    @abstractmethod
    def vectorize(self, texts: List[str]):
        pass


class TfIdfTextVectorizer(TextVectorizer):
    def __init__(self, max_features):
        self.tfidf_vec = TfidfVectorizer(max_features=max_features)

    def fit(self, texts: List[str]):
        self.tfidf_vec.fit(texts)
        os.makedirs(f'{SAVE_DIR}/tfidf', exist_ok=True)
        write_pickle(f'{SAVE_DIR}/tfidf/{VECTORIZER_NAME}', self.tfidf_vec)

    def vectorize(self, texts: List[str]):
        return self.tfidf_vec.transform(texts).toarray()


class EmbeddingTextVectorizer(TextVectorizer):
    def __init__(self, text_encoder: TextEncoderBase, embedding_dim, embeddings_loader: EmbeddingsLoader):
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


class LoadedTextVectorizer:
    @abstractmethod
    def vectorize(self, texts: List[str]):
        pass


class LoadedEmbeddingTextVectorizer(LoadedTextVectorizer):
    def __init__(self, text_encoder: LoadedTextEncoder, embedding_matrix):
        self.text_encoder = text_encoder
        self.embedding_matrix = embedding_matrix

    def vectorize(self, texts: List[str]):
        return self.text_encoder.encode(texts)


class LoadedTfIdfTextVectorizer(LoadedTextVectorizer):
    def __init__(self, vectorizer):
        self.tfidf_vec = vectorizer

    def vectorize(self, texts: List[str]):
        return self.tfidf_vec.transform(texts).toarray()
