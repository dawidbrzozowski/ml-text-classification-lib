from preprocessing.preprocessors import DataPreprocessor
from preprocessing.vectorization.data_vectorizers import DataVectorizer


def prepare_tfidf_train_test_data(data_params: dict, vectorizer_params: dict):
    data_extractor = data_params['data_extractor']()
    train_corpus, test_corpus = data_extractor.get_train_test_corpus()
    data_cleaner = data_params['data_cleaner']()
    text_vectorizer = data_params['text_vectorizer'](vectorizer_params['vector_width'])
    output_vectorizer = data_params['output_vectorizer']()
    data_vectorizer = DataVectorizer(text_vectorizer, output_vectorizer)
    preprocessor = DataPreprocessor(data_cleaner, data_vectorizer)
    preprocessor.fit(train_corpus)
    train_corpus = preprocessor.preprocess(train_corpus)['text_vectorized']
    test_corpus = preprocessor.preprocess(test_corpus)['text_vectorized']
    return train_corpus, test_corpus
