import re
from abc import abstractmethod
from typing import List
import en_core_web_sm

from utils.files_io import load_json

NER_CONVERTER_DEF_PATH = 'preprocessing/cleaning/resources/ner_converter.json'


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
    def __init__(self, replace_numbers=False, use_ner=False, use_ner_converter=False):
        self.replace_numbers = replace_numbers
        self.ner_tagger = en_core_web_sm.load() if use_ner else None
        self.ner_converter = None
        if use_ner and use_ner_converter:
            self.ner_converter = load_json(NER_CONVERTER_DEF_PATH)

    def clean(self, texts: List[str]):
        if self.ner_tagger:
            texts = self._perform_ner_on_texts(texts)
        if self.replace_numbers:
            texts = self._replace_numbers_with_symbol(texts)
        return texts

    def _perform_ner_on_texts(self, texts):
        processed_texts = []
        for text in texts:
            ents = self.ner_tagger(text).ents
            for ent in ents:
                convert = lambda label: label if self.ner_converter is None else self.ner_converter[label]
                text = text.replace(str(ent), convert(ent.label_))
            processed_texts.append(text)
        return processed_texts

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
