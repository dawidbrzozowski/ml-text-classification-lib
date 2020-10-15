from preprocessing.preprocessors import DataPreprocessor
from preprocessing.vectorization.data_vectorizers import DataVectorizer
from preprocessing.vectorization.embeddings.text_encoders import TextEncoder


def prepare_embedding_train_test_data(data_params: dict, vectorizer_params: dict):
    data_extractor = data_params['data_extractor']()
    train_corpus, test_corpus = data_extractor.get_train_test_corpus()
    data_cleaner = data_params['data_cleaner']()
    text_encoder = TextEncoder(
        max_vocab_size=vectorizer_params['max_vocab_size'],
        max_seq_len=vectorizer_params['max_seq_len'])
    text_vectorizer = data_params['text_vectorizer'](
        text_encoder=text_encoder,
        embedding_dim=vectorizer_params['embedding_dim'],
        embeddings_loader=data_params['embeddings_loader'](data_params['embedding_type']))
    output_vectorizer = data_params['output_vectorizer']()
    data_vectorizer = DataVectorizer(text_vectorizer, output_vectorizer)
    preprocessor = DataPreprocessor(data_cleaner, data_vectorizer)
    preprocessor.fit(train_corpus)
    train_corpus = preprocessor.preprocess(train_corpus)['text_vectorized']
    test_corpus = preprocessor.preprocess(test_corpus)['text_vectorized']
    return train_corpus, test_corpus
