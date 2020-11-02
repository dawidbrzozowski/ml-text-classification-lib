from abc import abstractmethod
from typing import List

import os

from sklearn.feature_extraction.text import TfidfVectorizer

from text_clsf_lib.preprocessing.vectorization.embeddings.embedding_loaders import EmbeddingsLoader
from text_clsf_lib.preprocessing.vectorization.embeddings.matrix_preparer import EmbeddingsMatrixPreparer
from text_clsf_lib.preprocessing.vectorization.embeddings.text_encoders import TextEncoderBase, LoadedTextEncoder
from utils.files_io import write_pickle, write_numpy
import numpy as np

VECTORIZER_NAME = 'vectorizer.vec'
EMBEDDING_MATRIX_NAME = 'embedding_matrix.npy'


class TextVectorizer:
    """
    Base class for TextVectorizers.
    """

    @abstractmethod
    def fit(self, texts: List[str]):
        pass

    @abstractmethod
    def vectorize(self, texts: List[str]):
        pass

    @abstractmethod
    def save(self, save_dir):
        pass


class TfIdfTextVectorizer(TextVectorizer):
    def __init__(self, max_features):
        self.tfidf_vec = TfidfVectorizer(max_features=max_features)

    def fit(self, texts: List[str]):
        self.tfidf_vec.fit(texts)

    def save(self, save_dir):
        os.makedirs(f'{save_dir}', exist_ok=True)
        write_pickle(f'{save_dir}/{VECTORIZER_NAME}', self.tfidf_vec)
        sorted_vocab = [item[0] for item in sorted(self.tfidf_vec.vocabulary_.items(), key=lambda item: item[1])]
        np.savetxt(f'{save_dir}/idf.txt', self.tfidf_vec.idf_, fmt='%f')
        with open(f'{save_dir}/vocab.txt', 'w') as f:
            for word in sorted_vocab:
                f.write(f'{word}\n')

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

    def save(self, save_dir):
        os.makedirs(f'{save_dir}', exist_ok=True)
        write_numpy(f'{save_dir}/{EMBEDDING_MATRIX_NAME}', self.embedding_matrix)
        self.text_encoder.save(save_dir)

    def vectorize(self, texts: List[str]):
        return self.text_encoder.encode(texts)


class LoadedTextVectorizer:
    """
    This is a base class for LoadedTextVectorizers.
    Their goal is to vectorize data, using preprocessing files created during training.
    """
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
