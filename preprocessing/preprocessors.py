from typing import List

from preprocessing.cleaning.data_cleaners import DataCleaner
from preprocessing.vectorization.data_vectorizers import DataVectorizer
from preprocessing.vectorization.text_vectorizers import LoadedTextVectorizer


class DataPreprocessor:
    """
    This class is meant to combine data cleaning and data vectorization.
    Together it should deliver the whole process of preprocessing the data.
    fit(...) should perform fitting for data cleaner and data vectorizer.
    preprocess(...) should clean the data first, and then vectorize.

    """
    def __init__(self, data_cleaner: DataCleaner, data_vectorizer: DataVectorizer):
        self.data_cleaner = data_cleaner
        self.data_vectorizer = data_vectorizer

    def fit(self, data: List[dict]):
        data = self.data_cleaner.clean(data)
        texts = [sample['text'] for sample in data]
        outputs = [sample['offensive'] for sample in data]
        self.data_vectorizer.fit(texts, outputs)

    def preprocess(self, data: List[dict]):
        self.data_cleaner.clean(data)
        texts = [record['text'] for record in data]
        outputs = [sample['offensive'] for sample in data]
        texts_vectorized, processed_outputs = self.data_vectorizer.vectorize(texts, outputs)
        return {
            'text_vectorized': texts_vectorized,
            'text': texts,
            'output': processed_outputs
        }


class RealDataPreprocessor:
    def __init__(self, text_cleaner: DataCleaner, loaded_text_vectorizer: LoadedTextVectorizer):
        # TODO for now it is DataCleaner. When Text and Output cleaner is implemented, change that to Text!
        self.text_cleaner = text_cleaner
        self.text_vectorizer = loaded_text_vectorizer

    def preprocess(self, data: str or List[str]):
        if type(data) is str:
            data = [data]
        data = self.text_cleaner.clean(data)
        return {
            'text_vectorized': self.text_vectorizer.vectorize(data),
            'text': data
        }
