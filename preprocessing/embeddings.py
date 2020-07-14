from typing import List

import numpy as np
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

from preprocessing.embedding_loaders import EmbeddingsLoader, GloveEmbeddingsLoader

class TextVectorizer:
    def __init__(self, max_vocab_size, max_seq_len):
        self.max_vocab_size = max_vocab_size
        self.max_seq_len = max_seq_len
        self.word2idx = None

    def fit_on_texts(self, texts: List[str]):
        pass

    def convert_texts_to_integers(self, texts: List[str]):
        pass


class TextWordVectorizer(TextVectorizer):
    def __init__(self, max_vocab_size, max_seq_len):
        super().__init__(max_vocab_size, max_seq_len)
        self.tokenizer = Tokenizer(num_words=max_vocab_size, lower=True)

    def fit_on_texts(self, texts):
        self.tokenizer.fit_on_texts(texts)
        self.word2idx = {word: idx for word, idx in self.tokenizer.word_index.items() if idx < self.tokenizer.num_words}

    def convert_texts_to_integers(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences=sequences, maxlen=self.max_seq_len)


class EmbeddingsMatrixPreparer:
    def __init__(self, word2idx, embedding_dim, embeddings_loader: EmbeddingsLoader = GloveEmbeddingsLoader()):
        self.word2idx = word2idx
        self.word2vec = embeddings_loader.load_word_vectors(embedding_dim)
        self.embedding_dim = embedding_dim

    def prepare_embedding_matrix(self):
        print('Filling pre-trained embeddings.')
        num_words = len(self.word2idx) + 1
        embedding_matrix = np.zeros((num_words, self.embedding_dim))
        for word, i in self.word2idx.items():
            embedding_vector = self.word2vec.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix
