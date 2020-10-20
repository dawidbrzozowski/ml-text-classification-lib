from typing import Tuple

from preprocessing.cleaning.data_cleaners import PresetDataCleaner, TextCleaner, OutputCleaner
from preprocessing.preprocessors import DataPreprocessor
from preprocessing.vectorization.data_vectorizers import DataVectorizer


def prepare_tfidf_train_test_data(data_params: dict, vectorizer_params: dict) -> dict:
    data_extractor = data_params['data_extractor']()
    train_corpus, test_corpus = data_extractor.get_train_test_corpus()
    text_cleaner = TextCleaner(**data_params['text_cleaning_params'])
    output_cleaner = OutputCleaner()
    data_cleaner = PresetDataCleaner(text_cleaner, output_cleaner)
    text_vectorizer = data_params['text_vectorizer'](vectorizer_params['vector_width'])
    output_vectorizer = data_params['output_vectorizer']()
    data_vectorizer = DataVectorizer(text_vectorizer, output_vectorizer)
    preprocessor = DataPreprocessor(data_cleaner, data_vectorizer)
    train_corpus = preprocessor.clean(train_corpus)
    test_corpus = preprocessor.clean(test_corpus)
    preprocessor.fit(train_corpus)
    train_corpus_vec = preprocessor.vectorize(train_corpus)
    test_corpus_vec = preprocessor.vectorize(test_corpus)
    return {
        'train_vectorized': train_corpus_vec,
        'test_vectorized': test_corpus_vec,
        'train_cleaned': train_corpus,
        'test_cleaned': test_corpus
    }
