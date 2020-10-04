from enum import Enum
from utils.files_io import load_json
import pandas as pd

LARGE_DATA_INPUT_DIR = 'data/large_data/input'
TASK_A_PATH = f'{LARGE_DATA_INPUT_DIR}/task_a_distant.tsv'
TASK_B_PATH = f'{LARGE_DATA_INPUT_DIR}/task_b_distant.tsv'
TASK_C_PATH = f'{LARGE_DATA_INPUT_DIR}/task_c_distant_ann.tsv'

BASELINE_DATA_DIR = 'data/unprocessed'


class DataType(Enum):
    TASK_A = TASK_A_PATH
    TASK_B = TASK_B_PATH
    TASK_C = TASK_C_PATH


class LargeDataExtractor:
    def __init__(self, data_type: DataType = DataType.TASK_A):
        self.data = self._load_data(data_type)

    def _load_data(self, data_type: DataType):
        return pd.read_csv(data_type.value, sep='\t')

    def process_n_rows(self, n: int):
        return self.data.iloc[:n, :].to_dict(orient='records')


class BaselineDataExtractor:

    def get_train_test_corpus(self, amount=1000000):
        data_train = load_json(f'{BASELINE_DATA_DIR}/{amount}/train_corpus.json')
        data_test = load_json(f'{BASELINE_DATA_DIR}/{amount}/test_corpus.json')
        return data_train, data_test
