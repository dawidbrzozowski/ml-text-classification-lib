import os
from argparse import ArgumentParser
from enum import Enum
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.files_io import write_json, load_json
from project_settings import RANDOM_STATE


LARGE_DATA_INPUT_DIR = 'data/large_data/input'
TASK_A_PATH = f'{LARGE_DATA_INPUT_DIR}/task_a_distant.tsv'
TASK_B_PATH = f'{LARGE_DATA_INPUT_DIR}/task_b_distant.tsv'
TASK_C_PATH = f'{LARGE_DATA_INPUT_DIR}/task_c_distant_ann.tsv'
MIN_AMOUNT = 1
MAX_AMOUNT = 8000000
DIR_NAME = 'data/unprocessed'
FILE_NAME = 'corpus.json'


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

def read_unprocessed_corpus(size: int):
    file_path = f'{DIR_NAME}/{size}/{FILE_NAME}'
    if os.path.exists(file_path):
        return load_json(file_path)
    else:
        return download_corpus(size)


def download_corpus(size: int, verbose=1):
    if verbose == 1:
        print(f'Performing download for {size} rows.')
    assert 0 < size < MAX_AMOUNT, f'Pick a number between {MIN_AMOUNT} and {MAX_AMOUNT}'
    data_extractor = SemevalDataRetriever()
    corpus = data_extractor.process_n_rows(size)
    return corpus


def save_downloaded_corpus(corpus, verbose=1):
    os.makedirs(f'{DIR_NAME}/{len(corpus)}', exist_ok=True)
    file_path = f'{DIR_NAME}/{len(corpus)}/{FILE_NAME}'
    write_json(file_path, corpus)
    if verbose == 1:
        print(f'Processed {len(corpus)} rows and saved to {file_path}')


def save_train_test_split(corpus, test_size: float, verbose=1):
    stratify_column = [row['offensive'] for row in corpus]
    if verbose == 1:
        print('Performing train test split...')
    train, test = train_test_split(corpus, stratify=stratify_column, test_size=test_size, random_state=RANDOM_STATE)
    write_json(f'{DIR_NAME}/{len(corpus)}/train_{FILE_NAME}', train)
    write_json(f'{DIR_NAME}/{len(corpus)}/test_{FILE_NAME}', test)
    if verbose == 1:
        print(f'Train test files saved to: {DIR_NAME}/{len(corpus)}')


def add_offensive_drop_avg_std(sample):
    sample['offensive'] = 1 if sample['average'] >= 0.5 else 0
    del sample['average']
    del sample['std']
    return sample


if __name__ == '__main__':
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--rows_amount', type=int, default=100000,
                                 help=f"Amount of rows to get. Should be number between {MIN_AMOUNT} and {MAX_AMOUNT}")
    argument_parser.add_argument('--verbose', type=int, default=1,
                                 help="Verbosity level. 1 for showing logs, 0 for silent.")
    argument_parser.add_argument('--test_size', type=float, default=0.2, help="Test size.")
    arguments = argument_parser.parse_args()
    corpus = download_corpus(arguments.rows_amount, verbose=arguments.verbose)
    corpus = [add_offensive_drop_avg_std(sample) for sample in corpus]
    save_downloaded_corpus(corpus, arguments.verbose)
    save_train_test_split(corpus, test_size=arguments.test_size, verbose=arguments.verbose)
