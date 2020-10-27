from text_clsf_lib.models.model_trainer_runner import NNModelRunner
from text_clsf_lib.preprocessing.cleaning.data_cleaners import TextCleaner
from text_clsf_lib.preprocessing.preprocessors import RealDataPreprocessor
from text_clsf_lib.preprocessing.vectorization.embeddings.text_encoders import LoadedTextEncoder
from text_clsf_lib.preprocessing.vectorization.text_vectorizers import LoadedEmbeddingTextVectorizer, LoadedTfIdfTextVectorizer
from utils.files_io import read_pickle, read_numpy


def get_embedding_preprocessor(preprocessing_params: dict):

    text_cleaner = TextCleaner(**preprocessing_params['text_cleaning_params'])
    vectorizer_params = preprocessing_params['vectorizer_params']
    max_seq_len = vectorizer_params['max_seq_len']
    embedding_matrix = read_numpy(vectorizer_params['embedding_matrix_path'])
    tokenizer = read_pickle(vectorizer_params['text_encoder_path'])
    text_encoder = LoadedTextEncoder(
        max_seq_len=max_seq_len,
        tokenizer=tokenizer)
    text_vectorizer = LoadedEmbeddingTextVectorizer(
        text_encoder=text_encoder,
        embedding_matrix=embedding_matrix)

    return RealDataPreprocessor(
        text_cleaner=text_cleaner,
        loaded_text_vectorizer=text_vectorizer)


def get_tfidf_preprocessor(preprocessing_params: dict):
    text_cleaner = TextCleaner(**preprocessing_params['text_cleaning_params'])
    vectorizer_params = preprocessing_params['vectorizer_params']
    vectorizer = read_pickle(vectorizer_params['vectorizer_path'])
    text_vectorizer = LoadedTfIdfTextVectorizer(vectorizer=vectorizer)

    return RealDataPreprocessor(
        text_cleaner=text_cleaner,
        loaded_text_vectorizer=text_vectorizer)


def get_model(model_path: str):
    return NNModelRunner(model_path=model_path)
