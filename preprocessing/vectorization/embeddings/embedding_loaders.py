from abc import abstractmethod

import numpy as np

EMBEDDINGS_DIR = 'preprocessing/vectorization/resources/embeddings/glove'


class EmbeddingsLoader:
    """
    This class is a base for Embedding loaders.
    It's goal is to return word2vec. (Dict with keys of type str and values ndarray of size embedding_dim)
    """
    def __init__(self, embedding_dir):
        self.embedding_dir = embedding_dir

    @abstractmethod
    def load_word_vectors(self, embedding_dim) -> dict:
        pass


class GloveEmbeddingsLoader(EmbeddingsLoader):
    def __init__(self, glove_type):
        embedding_dir = f'{EMBEDDINGS_DIR}/{glove_type}'
        super().__init__(embedding_dir)

    def load_word_vectors(self, embedding_dim) -> dict:
        print(f'Loading pre-trained GloVe word vectors from {self.embedding_dir}/{embedding_dim}d.txt.')
        word2vec = {}
        with open(f'{self.embedding_dir}/{embedding_dim}d.txt') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vec = np.asarray(values[1:], dtype='float32')
                word2vec[word] = vec
        print(f'Found {len(word2vec)} vectors.')
        return word2vec
