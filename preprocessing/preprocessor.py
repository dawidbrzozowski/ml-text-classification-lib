from typing import List

from preprocessing.data_extractor import LargeDataExtractor
from preprocessing.embedding_loaders import GloveEmbeddingsLoader
from preprocessing.embeddings import TextVectorizer, EmbeddingsMatrixPreparer, TextWordVectorizer

GLOVE_EMBEDDINGS_DIR = '../../large_files/glove.6B'


class TextPreprocessor:

    def fit(self, texts: List[str]):
        pass

    def preprocess(self, texts: List[str]):
        pass


class EmbeddingTextPreprocessor(TextPreprocessor):
    def __init__(self, text_vectorizer: TextVectorizer, embedding_dim, embeddings_loader=GloveEmbeddingsLoader()):
        self.text_vectorizer = text_vectorizer
        self.embedding_dim = embedding_dim
        self.embedding_matrix = None
        self.embeddings_loader = embeddings_loader

    def fit(self, texts: List[str]):
        # if preprocessing will make a lot of changes to the text
        # maybe it should be done before text vectorizer.
        self.text_vectorizer.fit_on_texts(texts)
        self.embedding_matrix = EmbeddingsMatrixPreparer(self.text_vectorizer.word2idx, self.embedding_dim,
                                                         self.embeddings_loader).prepare_embedding_matrix()

    def preprocess(self, texts: List[str]):
        # TODO text preprocessing part
        texts = [text.lower() for text in texts]

        return self.text_vectorizer.convert_texts_to_integers(texts)


class TfIdfTextPreprocessor(TextPreprocessor):
    def __init__(self):
        pass

    def preprocess(self, texts: List[str]):
        pass


class BasicOutputPreprocessor:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def preprocess(self, output: List[dict]):
        averages = [o['average'] for o in output]
        return [1 if average >= self.threshold else 0 for average in averages]


class SemEvalDataPreprocessor:
    def __init__(self, text_preprocessor: TextPreprocessor, output_preprocessor=BasicOutputPreprocessor()):
        self.text_preprocessor = text_preprocessor
        self.output_preprocessor = output_preprocessor

    def fit(self, data: List[dict]):
        texts = [record['text'] for record in data]
        self.text_preprocessor.fit(texts)

    def preprocess(self, data: List[dict]):
        texts = [record['text'] for record in data]
        outputs = [{'average': record['average'], 'std': record['std']} for record in data]
        processed_texts = self.text_preprocessor.preprocess(texts)
        processed_output = self.output_preprocessor.preprocess(outputs)
        return processed_texts, processed_output


data_extractor = LargeDataExtractor()

data = data_extractor.process_n_rows_to_dict(100000)
data_train = [record for i, record in enumerate(data) if i < 80000]
data_test = [record for i, record in enumerate(data) if i >= 80000]
text_vectorizer = TextWordVectorizer(max_vocab_size=5000, max_seq_len=50)
text_preprocessor = EmbeddingTextPreprocessor(text_vectorizer, embedding_dim=50)
data_preprocessor = SemEvalDataPreprocessor(text_preprocessor)
data_preprocessor.fit(data_train)
data_train = data_preprocessor.preprocess(data_train)
data_test = data_preprocessor.preprocess(data_test)