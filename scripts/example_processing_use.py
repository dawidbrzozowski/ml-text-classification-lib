from preprocessing.cleaning.data_cleaners import BaselineDataCleaner
from preprocessing.extraction.data_extractor import LargeDataExtractor
from preprocessing.vectorization.data_vectorizers import DataVectorizer
from preprocessing.vectorization.output_vectorizers import BasicOutputVectorizer
from preprocessing.preprocessors import DataPreprocessor
from preprocessing.vectorization.text_vectorizers import EmbeddingTextVectorizer
from preprocessing.vectorization.embeddings.text_encoders import TextEncoder

GLOVE_EMBEDDINGS_DIR = '../../large_files/glove.6B'


data_extractor = LargeDataExtractor()
data = data_extractor.process_n_rows_to_dict(100000)
data_train = [record for i, record in enumerate(data) if i < 80000]
data_test = [record for i, record in enumerate(data) if i >= 80000]
text_encoder = TextEncoder(max_vocab_size=5000, max_seq_len=50)
text_vectorizer = EmbeddingTextVectorizer(text_encoder, embedding_dim=50)
data_preprocessor = DataPreprocessor(BaselineDataCleaner(), DataVectorizer(text_vectorizer, BasicOutputVectorizer()))
data_preprocessor.fit(data_train)
data_train = data_preprocessor.preprocess(data_train)
data_test = data_preprocessor.preprocess(data_test)