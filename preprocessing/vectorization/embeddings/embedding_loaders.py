from abc import abstractmethod

import numpy as np

GLOVE_EMBEDDINGS_DIR = '../../large_files/glove.6B'


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
    def __init__(self):
        embedding_dir = GLOVE_EMBEDDINGS_DIR
        super().__init__(embedding_dir)

    def load_word_vectors(self, embedding_dim) -> dict:
        print('Loading pre-trained GloVe word vectors.')
        word2vec = {}
        with open(f'{self.embedding_dir}/glove.6B.{embedding_dim}d.txt') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vec = np.asarray(values[1:], dtype='float32')
                word2vec[word] = vec
        print(f'Found {len(word2vec)} vectors.')
        return word2vec
