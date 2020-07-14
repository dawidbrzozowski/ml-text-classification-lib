from typing import List

import numpy as np
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

from preprocessing.data_extractor import LargeDataExtractor

GLOVE_EMBEDDINGS_DIR = '../../large_files/glove.6B'


class EmbeddingsLoader:
    def __init__(self, embedding_dir):
        self.embedding_dir = embedding_dir

    def load_word_vectors(self, embedding_dim):
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


class TextVectorizer:
    """This class is meant to change text into sequence of integers that correspond to its word2vec representation."""

    def __init__(self, max_vocab_size, max_seq_len):
        self.max_vocab_size = max_vocab_size
        self.max_seq_len = max_seq_len
        self.tokenizer = Tokenizer(num_words=max_vocab_size, lower=True)
        self.word2idx = None

    def fit_on_texts(self, texts):
        self.tokenizer.fit_on_texts(texts)
        self.word2idx = {word: idx for word, idx in self.tokenizer.word_index.items() if idx < self.tokenizer.num_words}

    def convert_texts_to_integers(self, texts: List[str]):
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences=sequences, maxlen=self.max_seq_len)


class EmbeddingsPreparer:
    def __init__(self, max_vocab_size, max_seq_len, embedding_dim,
                 embeddings_loader: EmbeddingsLoader = GloveEmbeddingsLoader()):
        self.text_vectorizer = TextVectorizer(max_vocab_size, max_seq_len)
        self.word2vec = embeddings_loader.load_word_vectors(embedding_dim)
        self.embedding_dim = embedding_dim

    def prepare_embedding_matrix(self, texts):
        self.text_vectorizer.fit_on_texts(texts)
        print('Filling pre-trained embeddings.')
        num_words = len(self.text_vectorizer.word2idx) + 1
        embedding_matrix = np.zeros((num_words, self.embedding_dim))
        for word, i in self.text_vectorizer.word2idx.items():
            embedding_vector = self.word2vec.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def convert_texts_to_integers(self, texts: List[str]):
        return self.text_vectorizer.convert_texts_to_integers(texts)


data_extractor = LargeDataExtractor()

data = data_extractor.process_n_rows_to_dict(10000)
preparer = EmbeddingsPreparer(5000, 50, 50)
emb_matrix = preparer.prepare_embedding_matrix([d['text'] for d in data])
print(emb_matrix)