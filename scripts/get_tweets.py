import os
from argparse import ArgumentParser

from sklearn.model_selection import train_test_split

from text_clsf_lib.data_preparation.data_extracton import SemevalDataRetriever
from utils.files_io import write_json, load_json
from project_settings import RANDOM_STATE

MIN_AMOUNT = 1
MAX_AMOUNT = 8000000
STRATIFY_FRACTION = 0.1
DIR_NAME = 'data/unprocessed'
FILE_NAME = 'corpus.json'


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


def save_train_test_split(corpus, test_size: float, verbose=1, stratify_fraction=STRATIFY_FRACTION):
    if verbose == 1:
        print(f'Calculating stratify column for fraction: {stratify_fraction}...')
    stratify_column = [(row['average'] // stratify_fraction) for row in corpus]
    if verbose == 1:
        print('Performing train test split...')
    train, test = train_test_split(corpus, stratify=stratify_column, test_size=test_size, random_state=RANDOM_STATE)
    write_json(f'{DIR_NAME}/{len(corpus)}/train_{FILE_NAME}', train)
    write_json(f'{DIR_NAME}/{len(corpus)}/test_{FILE_NAME}', test)
    if verbose == 1:
        print(f'Train test files saved to: {DIR_NAME}/{len(corpus)}')


if __name__ == '__main__':
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--rows_amount', type=int, default=100000,
                                 help=f"Amount of rows to get. Should be number between {MIN_AMOUNT} and {MAX_AMOUNT}")
    argument_parser.add_argument('--verbose', type=int, default=1,
                                 help="Verbosity level. 1 for showing logs, 0 for silent.")
    argument_parser.add_argument('--test_size', type=float, default=0.2, help="Test size.")
    arguments = argument_parser.parse_args()
    corpus = download_corpus(arguments.rows_amount, verbose=arguments.verbose)
    save_downloaded_corpus(corpus, arguments.verbose)
    save_train_test_split(corpus, test_size=arguments.test_size, verbose=arguments.verbose)
