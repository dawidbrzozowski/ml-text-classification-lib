from preprocessing.preprocessors import DataPreprocessor
from preprocessing.vectorization.data_vectorizers import DataVectorizer


def prepare_train_test_data(preset: dict):
    data_extractor = preset['data_extractor']()
    train_corpus, test_corpus = data_extractor.get_train_test_corpus()
    data_cleaner = preset['data_cleaner']()
    text_vectorizer = preset['text_vectorizer']()
    output_vectorizer = preset['output_vectorizer']()
    data_vectorizer = DataVectorizer(text_vectorizer, output_vectorizer)
    preprocessor = DataPreprocessor(data_cleaner, data_vectorizer)
    preprocessor.fit(train_corpus)
    train_corpus = preprocessor.preprocess(train_corpus)
    test_corpus = preprocessor.preprocess(test_corpus)
    return train_corpus, test_corpus
