from text_clsf_lib.preprocessing.vectorization.data_vectorizers import DataVectorizer
from text_clsf_lib.preprocessing.vectorization.embeddings.text_encoders import TextEncoder


def prepare_embedding_data_vectorizer(vectorizer_params):
    """
    Takes parameters from preset vectorizer_params and prepares custom DataVectorizer based on embedding.
    :param vectorizer_params: dict
    :return: DataVectorizer based on embedding architecture.
    """
    text_encoder = TextEncoder(
        max_vocab_size=vectorizer_params['max_vocab_size'],
        max_seq_len=vectorizer_params['max_seq_len'])

    text_vectorizer = vectorizer_params['text_vectorizer'](
        text_encoder=text_encoder,
        embedding_dim=vectorizer_params['embedding_dim'],
        embeddings_loader=vectorizer_params['embeddings_loader'](vectorizer_params['embedding_type']))

    output_vectorizer = vectorizer_params['output_vectorizer']()
    return DataVectorizer(text_vectorizer, output_vectorizer)