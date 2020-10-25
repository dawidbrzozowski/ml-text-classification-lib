from preprocessing.vectorization.data_vectorizers import DataVectorizer


def prepare_tfidf_data_vectorizer(vectorizer_params):
    vector_width = vectorizer_params['vector_width']
    text_vectorizer = vectorizer_params['text_vectorizer'](vector_width)
    output_vectorizer = vectorizer_params['output_vectorizer']()
    return DataVectorizer(text_vectorizer, output_vectorizer)