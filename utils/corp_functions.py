BASELINE_DATA_DIR = 'data/unprocessed'


def generate_semeval_load_data_args(num_samples: int):
    train_path = f'{BASELINE_DATA_DIR}/{num_samples}/train_corpus.json'
    test_path = f'{BASELINE_DATA_DIR}/{num_samples}/test_corpus.json'
    path = (train_path, test_path)
    return {
        'data_path': path,
        'X_name': 'text',
        'y_name': 'offensive',
    }
