from abc import abstractmethod
from typing import List

from preprocessing.vectorization.embeddings.embedding_loaders import GloveEmbeddingsLoader
from preprocessing.vectorization.embeddings.embeddings import EmbeddingsMatrixPreparer
from preprocessing.vectorization.embeddings.text_encoders import TextEncoderBase


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


class TfIdfTextVectorizer(TextVectorizer):

    def fit(self, texts: List[str]):
        pass

    def vectorize(self, texts: List[str]):
        pass


class EmbeddingTextVectorizer(TextVectorizer):
    def __init__(self, text_encoder: TextEncoderBase, embedding_dim, embeddings_loader=GloveEmbeddingsLoader()):
        self.text_encoder = text_encoder
        self.embedding_dim = embedding_dim
        self.embedding_matrix = None
        self.embeddings_loader = embeddings_loader

    def fit(self, texts: List[str]):
        self.text_encoder.fit(texts)
        word2vec = self.embeddings_loader.load_word_vectors(self.embedding_dim)
        emb_matrix_preparer = EmbeddingsMatrixPreparer(self.text_encoder.word2idx, self.embedding_dim, word2vec)
        self.embedding_matrix = emb_matrix_preparer.prepare_embedding_matrix()

    def vectorize(self, texts: List[str]):
        encoded = self.text_encoder.encode(texts)
        return encoded

    def get_embedding_matrix(self):
        return self.embedding_matrix
