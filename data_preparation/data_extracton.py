from abc import abstractmethod
from enum import Enum

from project_settings import RANDOM_STATE
from utils.files_io import load_json
import pandas as pd
from sklearn.model_selection import train_test_split

LARGE_DATA_INPUT_DIR = 'data/large_data/input'
TASK_A_PATH = f'{LARGE_DATA_INPUT_DIR}/task_a_distant.tsv'
TASK_B_PATH = f'{LARGE_DATA_INPUT_DIR}/task_b_distant.tsv'
TASK_C_PATH = f'{LARGE_DATA_INPUT_DIR}/task_c_distant_ann.tsv'

BASELINE_DATA_DIR = 'data/unprocessed'


class DataType(Enum):
    TASK_A = TASK_A_PATH
    TASK_B = TASK_B_PATH
    TASK_C = TASK_C_PATH


class SemevalDataRetriever:
    def __init__(self, data_type: DataType = DataType.TASK_A):
        self.data = self._load_data(data_type)

    def _load_data(self, data_type: DataType):
        return pd.read_csv(data_type.value, sep='\t')

    def process_n_rows(self, n: int):
        return self.data.iloc[:n, :].to_dict(orient='records')


class DataExtractor:
    @abstractmethod
    def get_train_test_corpus(self, **kwargs):
        pass


class BaselineJsonDataExtractor(DataExtractor):

    def get_train_test_corpus(self, amount=1000000):
        data_train = load_json(f'{BASELINE_DATA_DIR}/{amount}/train_corpus.json')
        data_test = load_json(f'{BASELINE_DATA_DIR}/{amount}/test_corpus.json')
        return data_train, data_test


class CustomPathJsonDataExtractor(DataExtractor):
    """
    The provided Json must be a list of dict containing fields such as:
    - 'text'
    - (optional) id
    - offensive (0 or 1) OR (true or false)
    """

    def get_train_test_corpus(self, train_path, test_path):
        data_train = load_json(train_path)
        data_test = load_json(test_path)
        data_train = list(map(change_to_number, data_train))
        data_test = list(map(change_to_number, data_test))

        return data_train, data_test


# abstract class
class SingleFileDataExtractor(DataExtractor):

    def get_train_test_corpus(self, corpus, test_size):
        stratify_column = [sample['offensive'] for sample in corpus]
        data_train, data_test = train_test_split(
            corpus, stratify=stratify_column, test_size=test_size, random_state=RANDOM_STATE)
        data_train = list(map(change_to_number, data_train))
        data_test = list(map(change_to_number, data_test))
        return data_train, data_test


class CustomPathSingleFileJsonDataExtractor(SingleFileDataExtractor):
    """
    The provided Json must be a list of dict containing fields such as:
    - 'text'
    - (optional) id
    - offensive (0 or 1) OR (true or false)
    """

    def get_train_test_corpus(self, corpus_path, test_size=0.2):
        corpus = load_json(corpus_path)
        return super().get_train_test_corpus(corpus, test_size)


class TxtDataExtractor(DataExtractor):

    @abstractmethod
    def get_train_test_corpus(self, **kwargs):
        pass

    def read_txt_corp(self, path, delimiter):
        corp = []
        mapper = {
            '1': 1,
            '0': 0,
            'true': 1,
            'false': 0
        }
        with open(path, 'r') as tr_file:
            for line in tr_file:
                values = line.rstrip('\n').split(delimiter)
                assert len(values) == 3, 'Each line should contain 2 delimiters.'
                corp.append({
                    'id': int(values[0]),
                    'text': values[1],
                    'offensive': mapper[values[2]] # get rid of newline newline
                })
        return corp


class CustomPathTxtDataExtractor(TxtDataExtractor):
    """
    Each sample in different line.
    Each line:
    Id; Text; Offensive
    """

    def get_train_test_corpus(self, train_path, test_path, delimiter=';'):
        train_corp = self.read_txt_corp(train_path, delimiter)
        test_corp = self.read_txt_corp(test_path, delimiter)
        return train_corp, test_corp


class SingleFileCustomPathTxtDataExtractor(SingleFileDataExtractor, TxtDataExtractor):
    def get_train_test_corpus(self, corpus_path, test_size=0.2, delimiter=';'):
        corpus = self.read_txt_corp(corpus_path, delimiter)
        return super().get_train_test_corpus(corpus, test_size)


def change_to_number(sample):
    sample['offensive'] = 1 if sample['offensive'] else 0
    return sample
