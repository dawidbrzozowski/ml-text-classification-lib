import unittest

from text_clsf_lib.data_preparation.data_extracton import SemevalDataRetriever, DataType


class DataRetrievalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_extractor_a = SemevalDataRetriever(DataType.TASK_A)

    def test_len_extraction_to_dict(self):
        extracted_dict = self.data_extractor_a.process_n_rows(100)
        self.assertEqual(len(extracted_dict), 100)

    def test_columns_extracted_to_dict(self):
        extracted_dict = self.data_extractor_a.process_n_rows(100)
        self.assertEqual(list(extracted_dict[0].keys()), ['id', 'text', 'average', 'std'])