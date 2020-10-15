import re
from abc import abstractmethod
from typing import List


class DataCleaner:
    """
    This class is meant to be responsible for cleaning text data and output data.
    It should remove damaged records and process the rest.
    TODO It should be split on two classes just like in vectorizer, where each one is responsible for text and output.
    """

    @abstractmethod
    def clean(self, data: List[dict]):
        pass


class BaselineDataCleaner(DataCleaner):
    def clean(self, data: List[dict]):
        return data


class TextCleaner:
    def clean(
            self,
            texts: List[str],
            replace_numbers=False):
        if replace_numbers:
            texts = self._replace_numbers_with_symbol(texts)
        return texts

    def _replace_numbers_with_symbol(self, texts):
        number_pattern = re.compile(r'\d+')
        return [number_pattern.sub('number_encoded', text) for text in texts]


class OutputCleaner:
    """
    Output cleaner will remove rows, for which output is invalid.
    """
    def clean(self, data: List[dict]):
        correct_data = []
        for sample in data:
            if sample['offensive'] in (0, 1):
                correct_data.append(sample)
        return correct_data


class PresetDataCleaner(DataCleaner):
    def __init__(self, text_cleaner: TextCleaner, output_cleaner: OutputCleaner):
        self.text_cleaner = text_cleaner
        self.output_cleaner = output_cleaner

    def clean(self, data: List[dict]):
        texts = [sample['text'] for sample in data]
        cleaned_texts = self.text_cleaner.clean(texts)
        for sample, cleaned_text in zip(data, cleaned_texts):
            sample['text'] = cleaned_text
        return self.output_cleaner.clean(data)

