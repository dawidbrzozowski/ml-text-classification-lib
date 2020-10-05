from models.model_commons import NNModelRunner
from preprocessing.preprocessors import RealDataPreprocessor
from preprocessing.vectorization.embeddings.text_encoders import TextEncoder
from preprocessing.vectorization.text_vectorizers import EmbeddingTextVectorizer
from utils.files_io import read_pickle, read_numpy


def get_embedding_preprocessor(preset: dict):
    data_cleaner = preset['data_params']['data_cleaner']()
    max_vocab_size = preset['vectorizer_params']['max_vocab_size']
    max_seq_len = preset['vectorizer_params']['max_seq_len']
    embedding_dim = preset['vectorizer_params']['embedding_dim']
    embedding_matrix = read_numpy(preset['vectorizer_params']['embedding_matrix_path'])
    tokenizer = read_pickle(preset['vectorizer_params']['text_encoder_path'])
    text_encoder = TextEncoder(
        max_vocab_size=max_vocab_size,
        max_seq_len=max_seq_len,
        tokenizer=tokenizer)
    embedding_loader = preset['vectorizer_params']['embedding_loader']()
    text_vectorizer = EmbeddingTextVectorizer(
        text_encoder=text_encoder,
        embedding_dim=embedding_dim,
        embeddings_loader=embedding_loader,
        embedding_matrix=embedding_matrix)

    return RealDataPreprocessor(
        ready_text_cleaner=data_cleaner,
        ready_text_vectorizer=text_vectorizer)


def get_model(preset: dict):
    return NNModelRunner(model_path=preset['model_path'])
