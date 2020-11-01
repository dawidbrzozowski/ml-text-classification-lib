import unittest
from text_clsf_lib.data_preparation.data_extracton import CustomPathJsonDataExtractor, BaselineJsonDataExtractor, \
    CustomPathSingleFileJsonDataExtractor, SingleFileCustomPathTxtDataExtractor, CustomPathTxtDataExtractor

DATA_EXTRACTOR_TEST_DIR = 'tests/resources/data_extractor_tests'


class DataExtractionTest(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.required_fields = ('text', 'offensive')

    def test_json_train_test_baseline(self):
        extr = BaselineJsonDataExtractor()
        data_tr, data_te = extr.get_train_test_corpus()
        req_fields = ('text', 'offensive')
        for field in req_fields:
            self.assertIn(field, data_tr[0].keys())

    def prepare_test_for_train_test_extr(self, extractor, train_path, test_path, expected_offensive_type):
        extr = extractor()
        data_tr, data_te = extr.get_train_test_corpus(train_path=train_path, test_path=test_path)

        sample = data_tr[0]

        for field in self.required_fields:
            self.assertIn(field, sample.keys())

        self.assertIs(type(sample['offensive']), expected_offensive_type)

    def prepare_test_for_corpus_extr(self, extractor, corpus_path, expected_offensive_type):
        extr = extractor()
        data_tr, data_te = extr.get_train_test_corpus(corpus_path=corpus_path, test_size=0.5)

        sample = data_tr[0]

        for field in self.required_fields:
            self.assertIn(field, sample.keys())

        self.assertIs(type(sample['offensive']), expected_offensive_type)

    def test_json_train_test_bool(self):
        self.prepare_test_for_train_test_extr(
            extractor=CustomPathJsonDataExtractor,
            train_path=f'{DATA_EXTRACTOR_TEST_DIR}/train_test_separate/json_train_bool.json',
            test_path=f'{DATA_EXTRACTOR_TEST_DIR}/train_test_separate/json_test_bool.json',
            expected_offensive_type=int)

    def test_json_train_test_number(self):
        self.prepare_test_for_train_test_extr(
            extractor=CustomPathJsonDataExtractor,
            train_path=f'{DATA_EXTRACTOR_TEST_DIR}/train_test_separate/json_train_number.json',
            test_path=f'{DATA_EXTRACTOR_TEST_DIR}/train_test_separate/json_test_number.json',
            expected_offensive_type=int)

    def test_json_corpus_bool(self):
        self.prepare_test_for_corpus_extr(
            extractor=CustomPathSingleFileJsonDataExtractor,
            corpus_path=f'{DATA_EXTRACTOR_TEST_DIR}/one_corpus/json_bool.json',
            expected_offensive_type=int)

    def test_json_corpus_number(self):
        self.prepare_test_for_corpus_extr(
            extractor=CustomPathSingleFileJsonDataExtractor,
            corpus_path=f'{DATA_EXTRACTOR_TEST_DIR}/one_corpus/json_number.json',
            expected_offensive_type=int)

    def test_txt_corpus_bool(self):
        self.prepare_test_for_corpus_extr(
            extractor=SingleFileCustomPathTxtDataExtractor,
            corpus_path=f'{DATA_EXTRACTOR_TEST_DIR}/one_corpus/txt_bool.txt',
            expected_offensive_type=int)

    def test_txt_corpus_number(self):
        self.prepare_test_for_corpus_extr(
            extractor=SingleFileCustomPathTxtDataExtractor,
            corpus_path=f'{DATA_EXTRACTOR_TEST_DIR}/one_corpus/txt_number.txt',
            expected_offensive_type=int)

    def test_txt_train_test_bool(self):
        self.prepare_test_for_train_test_extr(
            extractor=CustomPathTxtDataExtractor,
            train_path=f'{DATA_EXTRACTOR_TEST_DIR}/train_test_separate/txt_train_bool.txt',
            test_path=f'{DATA_EXTRACTOR_TEST_DIR}/train_test_separate/txt_test_bool.txt',
            expected_offensive_type=int)

    def test_txt_train_test_number(self):
        self.prepare_test_for_train_test_extr(
            extractor=CustomPathTxtDataExtractor,
            train_path=f'{DATA_EXTRACTOR_TEST_DIR}/train_test_separate/txt_train_number.txt',
            test_path=f'{DATA_EXTRACTOR_TEST_DIR}/train_test_separate/txt_test_number.txt',
            expected_offensive_type=int)


if __name__ == '__main__':
    unittest.main()
