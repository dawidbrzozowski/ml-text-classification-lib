from enum import Enum

import pandas as pd

LARGE_DATA_INPUT_DIR = 'large_data/input'
TASK_A_PATH = f'{LARGE_DATA_INPUT_DIR}/task_a_distant.tsv'
TASK_B_PATH = f'{LARGE_DATA_INPUT_DIR}/task_b_distant.tsv'
TASK_C_PATH = f'{LARGE_DATA_INPUT_DIR}/task_c_distant_ann.tsv'


class DataType(Enum):
    TASK_A = TASK_A_PATH
    TASK_B = TASK_B_PATH
    TASK_C = TASK_C_PATH


class LargeDataExtractor:
    def __init__(self, data_type: DataType):
        self.data = self._load_data(data_type)

    def _load_data(self, data_type: DataType):
        return pd.read_csv(data_type.value, sep='\t')

    def process_n_rows_to_dict(self, n: int):
        return self.data.iloc[:n, :].to_dict(orient='records')
