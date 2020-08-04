from configs.config_appliers import Config
from preprocessing.extraction.data_extractor import LargeDataExtractor


GLOVE_EMBEDDINGS_DIR = '../../large_files/glove.6B'


data_extractor = LargeDataExtractor()
data = data_extractor.process_n_rows_to_dict(100000)
data_train = [record for i, record in enumerate(data) if i < 80000]
data_test = [record for i, record in enumerate(data) if i >= 80000]
config = Config('configs/data/model_config.json')
data_preprocessor = config.get_preprocessor()
data_preprocessor.fit(data_train)
data_train = data_preprocessor.preprocess(data_train)