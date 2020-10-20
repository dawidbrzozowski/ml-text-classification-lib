from models.model_trainer_runner import NNModelRunner
from preprocessing.cleaning.data_cleaners import TextCleaner
from preprocessing.preprocessors import RealDataPreprocessor
from preprocessing.vectorization.embeddings.text_encoders import LoadedTextEncoder
from preprocessing.vectorization.text_vectorizers import LoadedEmbeddingTextVectorizer
from utils.files_io import read_pickle, read_numpy


def get_embedding_preprocessor(preset: dict):
    text_cleaner = TextCleaner(**preset['data_params']['text_cleaning_params'])
    max_seq_len = preset['vectorizer_params']['max_seq_len']
    embedding_matrix = read_numpy(preset['vectorizer_params']['embedding_matrix_path'])
    tokenizer = read_pickle(preset['vectorizer_params']['text_encoder_path'])
    text_encoder = LoadedTextEncoder(
        max_seq_len=max_seq_len,
        tokenizer=tokenizer)
    text_vectorizer = LoadedEmbeddingTextVectorizer(
        text_encoder=text_encoder,
        embedding_matrix=embedding_matrix)

    return RealDataPreprocessor(
        text_cleaner=text_cleaner,
        loaded_text_vectorizer=text_vectorizer)


def get_tfidf_preprocessor(preset: dict):
    text_cleaner = TextCleaner(**preset['data_params']['text_cleaning_params'])
    vectorizer = read_pickle(preset['vectorizer_params']['vectorizer_path'])
    text_vectorizer = preset['data_params']['text_vectorizer'](
        vectorizer=vectorizer)

    return RealDataPreprocessor(
        text_cleaner=text_cleaner,
        loaded_text_vectorizer=text_vectorizer)


def get_model(preset: dict):
    return NNModelRunner(model_path=preset['model_path'])
