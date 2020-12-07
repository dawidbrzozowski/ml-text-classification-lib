from abc import abstractmethod
from enum import Enum
from typing import List, Tuple

from project_settings import RANDOM_STATE
from utils.files_io import load_json
import pandas as pd
from sklearn.model_selection import train_test_split

LARGE_DATA_INPUT_DIR = 'data/large_data/input'
TASK_A_PATH = f'{LARGE_DATA_INPUT_DIR}/task_a_distant.tsv'
TASK_B_PATH = f'{LARGE_DATA_INPUT_DIR}/task_b_distant.tsv'
TASK_C_PATH = f'{LARGE_DATA_INPUT_DIR}/task_c_distant_ann.tsv'

BASELINE_DATA_DIR = 'data/unprocessed'

# TODO get_train_test_corpus() musi być bezargumentowy.
# jedyne co to konstruktor powinien wszystko wczytać
class DataType(Enum):
    """
    Holds paths to different kind of SemEval tasks.

    Task A: Offensive language identification
    Task B: Automatic categorization of offense types
    Task C: Offense target identification

    More info: https://www.aclweb.org/anthology/S19-2010.pdf
    """
    TASK_A = TASK_A_PATH
    TASK_B = TASK_B_PATH
    TASK_C = TASK_C_PATH


class SemevalDataRetriever:
    """
    This class is meant to retrieve a smaller chunk of data from the whole ~8,000,000 records.
    """

    def __init__(self, data_type: DataType = DataType.TASK_A):
        self.data = self._load_data(data_type)

    def _load_data(self, data_type: DataType):
        return pd.read_csv(data_type.value, sep='\t')

    def process_n_rows(self, n: int) -> List[dict]:
        """
        :param n: Amount of rows to return.
        :return: List of size n.
        Each element is a dict with id, text, avg and std.
        """

        return self.data.iloc[:n, :].to_dict(orient='records')


class DataExtractor:
    """
    Base class for retrieving train test sets.
    """

    @abstractmethod
    def get_train_test_corpus(self, **kwargs) -> Tuple[List, List]:
        pass


class BaselineJsonDataExtractor(DataExtractor):
    """
    Retrieves data from BASELINE_DATA_DIR. This data comes from SemEval competition.
    """

    def get_train_test_corpus(self, amount=5000000) -> Tuple[List, List]:
        data_train = load_json(f'{BASELINE_DATA_DIR}/{amount}/train_corpus.json')
        data_test = load_json(f'{BASELINE_DATA_DIR}/{amount}/test_corpus.json')
        data_train = list(map(add_offensive_drop_avg_std, data_train))
        data_test = list(map(add_offensive_drop_avg_std, data_test))

        return data_train, data_test


class CustomPathJsonDataExtractor(DataExtractor):
    """
    This class is meant to retrieve data from custom jsons (train and test separately).

    The provided Json must be a list of dict containing fields such as:
    - 'text': str
    - 'offensive' (0 or 1) OR (true or false)
    """

    def get_train_test_corpus(self, train_path, test_path) -> Tuple[List, List]:
        """

        :param train_path: path to train set.
        :param test_path: path to test set
        :return: List[dict] train, List[dict] test.
        """
        data_train = load_json(train_path)
        data_test = load_json(test_path)
        data_train = list(map(change_to_number, data_train))
        data_test = list(map(change_to_number, data_test))

        return data_train, data_test


class SingleFileDataExtractor(DataExtractor):
    """
    Base class for retrieving data from custom file (train and test together).
    The split is performed on random state from project settings.
    """

    def get_train_test_corpus(self, corpus: List[dict], test_size: float):
        stratify_column = [sample['offensive'] for sample in corpus]
        data_train, data_test = train_test_split(
            corpus, stratify=stratify_column, test_size=test_size, random_state=RANDOM_STATE)
        data_train = list(map(change_to_number, data_train))
        data_test = list(map(change_to_number, data_test))
        return data_train, data_test


class CustomPathSingleFileJsonDataExtractor(SingleFileDataExtractor):
    """
    This class is meant to retrieve data from custom json (train and test together).

    The provided Json must be a list of dict containing fields such as:
    - 'text': str
    - 'offensive' (0 or 1) OR (true or false)
    """

    def get_train_test_corpus(self, corpus_path: str, test_size=0.2):
        """
        :param corpus_path: Path to corpus.
        :param test_size: float
        :return: List[dict] train, List[dict] test.
        """
        corpus = load_json(corpus_path)
        return super().get_train_test_corpus(corpus, test_size)


class TxtDataExtractor(DataExtractor):
    """
       Base class for retrieving data from custom .txt file (train and test together).

       The provided txt file must contain be a list of dict containing fields such as:
       - 'text': str
       - 'offensive' (0 or 1) OR (true or false)
       """

    @abstractmethod
    def get_train_test_corpus(self, **kwargs):
        pass

    def read_txt_corp(self, path: str, delimiter):
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
                assert len(values) == 2, 'Each line should contain 1 delimiter.'
                corp.append({
                    'text': values[0],
                    'offensive': mapper[values[1]]  # get rid of newline newline
                })
        return corp


class CustomPathTxtDataExtractor(TxtDataExtractor):
    """
    This class is meant to retrieve data from custom .txt files (train and test separately).

    Each sample should be in different line.

    Each line should look like this:
    Text; Offensive
    """

    def get_train_test_corpus(self, train_path, test_path, delimiter=';'):
        train_corp = self.read_txt_corp(train_path, delimiter)
        test_corp = self.read_txt_corp(test_path, delimiter)
        return train_corp, test_corp


class SingleFileCustomPathTxtDataExtractor(SingleFileDataExtractor, TxtDataExtractor):
    """
       This class is meant to retrieve data from custom .txt file (train and test together).

       Each sample should be in different line.

       Each line should look like this:
       Text; Offensive
       """
    def get_train_test_corpus(self, corpus_path, test_size=0.2, delimiter=';'):
        corpus = self.read_txt_corp(corpus_path, delimiter)
        return super().get_train_test_corpus(corpus, test_size)


def change_to_number(sample):
    sample['offensive'] = 1 if sample['offensive'] else 0
    return sample


def add_offensive_drop_avg_std(sample):
    sample['offensive'] = 1 if sample['average'] >= 0.5 else 0
    del sample['average']
    del sample['std']
    return sample
